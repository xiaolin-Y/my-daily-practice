import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import time
from mpl_toolkits.mplot3d import Axes3D

# ==================== 1. 参数定义 ====================
class BeamParameters:
    def __init__(self):
        self.L = 10.0          # 梁长度(m)
        self.D = 1.0           # 梁高度(m)
        self.E = 2.1e5           # 弹性模量(Pa)
        self.nu = 0.3          # 泊松比
        self.t = 1.0           # 厚度(m)
        self.Py = -1.0         # 自由端集中力(N)
        self.I = self.t * self.D**3 / 12  # 截面惯性矩(m^4)
        self.scale_factor = 50 # 位移可视化放大系数

# ==================== 2. 解析解函数 ====================
def analytical_solution(x, y, params):
    """Timoshenko梁理论解析解"""
    term1 = (6*params.L - 3*x)*x
    term2 = (2 + params.nu)*(y**2 - params.D**2/4)
    u = -params.Py*y/(6*params.E*params.I) * (term1 + term2)
    
    term3 = 3*params.nu*y**2*(params.L - x)
    term4 = (4 + 5*params.nu)*params.D**2*x/4
    term5 = (3*params.L - x)*x**2
    v = params.Py/(6*params.E*params.I) * (term3 + term4 + term5)
    return u, v

# ==================== 3. 网格生成 ====================
def generate_mesh(params, nx, ny):
    """生成结构化三角形网格"""
    x = np.linspace(0, params.L, nx)
    y = np.linspace(-params.D/2, params.D/2, ny)
    X, Y = np.meshgrid(x, y)
    nodes = np.column_stack((X.flatten(), Y.flatten()))
    
    elements = []
    for j in range(ny-1):
        for i in range(nx-1):
            n1 = j*nx + i
            n2 = j*nx + i+1
            n3 = (j+1)*nx + i
            n4 = (j+1)*nx + i+1
            elements.append([n1, n2, n3])
            elements.append([n2, n4, n3])
    
    return nodes, np.array(elements)

# ==================== 4. 有限元计算 ====================
def fem_solution(nodes, elements, params):
    """执行有限元计算"""
    nnodes = nodes.shape[0]
    ndof = 2 * nnodes
    
    K = assemble_stiffness_matrix(nodes, elements, params)
    
    U = np.zeros(ndof)
    F = np.zeros(ndof)
    
    fixed_nodes = [i for i in range(nnodes) if np.isclose(nodes[i,0], 0)]
    fixed_dofs = []
    for n in fixed_nodes:
        u, v = analytical_solution(nodes[n,0], nodes[n,1], params)
        U[2*n] = u
        U[2*n+1] = v
        fixed_dofs.extend([2*n, 2*n+1])
    
    free_end_nodes = [i for i in range(nnodes) if np.isclose(nodes[i,0], params.L)]
    if free_end_nodes:
        force_node = free_end_nodes[np.argmin(np.abs(nodes[free_end_nodes,1]))]
        F[2*force_node+1] = params.Py
    
    free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)
    K_free = K[free_dofs,:][:,free_dofs]
    F_eff = F[free_dofs] - K[free_dofs,:][:,fixed_dofs] @ U[fixed_dofs]
    
    U[free_dofs] = spsolve(K_free, F_eff)
    
    return U



def assemble_stiffness_matrix(nodes, elements, params):
    """组装刚度矩阵"""
    nnodes = nodes.shape[0]
    K = lil_matrix((2*nnodes, 2*nnodes))
    
    D = params.E/(1-params.nu**2) * np.array([
        [1, params.nu, 0],
        [params.nu, 1, 0],
        [0, 0, (1-params.nu)/2]
    ])
    
    for elem in elements:
        x = nodes[elem, 0]
        y = nodes[elem, 1]
        
        Ke = triangle_stiffness_matrix(x, y, D, params.t)
        
        dofs = np.array([2*elem[0], 2*elem[0]+1,
                        2*elem[1], 2*elem[1]+1,
                        2*elem[2], 2*elem[2]+1]).flatten()
        
        for i in range(6):
            for j in range(6):
                K[dofs[i], dofs[j]] += Ke[i,j]
    
    return csr_matrix(K)

def triangle_stiffness_matrix(x, y, D, t):
    """三角形单元刚度矩阵"""
    area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (y[1]-y[0])*(x[2]-x[0]))
    
    b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
    c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
    
    B = np.zeros((3,6))
    B[0,0::2] = b
    B[1,1::2] = c
    B[2,0::2] = c
    B[2,1::2] = b
    B /= (2*area)
    
    return t * area * B.T @ D @ B

# ==================== 5. 结果可视化 ====================
def plot_results(nodes, elements, U_fem, params):
    """绘制所有结果图形"""
    plt.close('all')
    
    # 准备数据
    tri = Triangulation(nodes[:,0], nodes[:,1], elements)
    u_fem = U_fem[::2]
    v_fem = U_fem[1::2]
    
    u_ana = np.zeros_like(u_fem)
    v_ana = np.zeros_like(v_fem)
    for i, (x, y) in enumerate(nodes):
        u_ana[i], v_ana[i] = analytical_solution(x, y, params)
    
    u_error = np.abs(u_fem - u_ana)
    v_error = np.abs(v_fem - v_ana)
    total_error = np.sqrt(u_error**2 + v_error**2)
    deformed_nodes = nodes + params.scale_factor * np.column_stack((u_fem, v_fem))
    free_end = np.isclose(nodes[:,0], params.L)
    
    # ========== 第一个Figure：2D图 ==========
    fig1 = plt.figure(figsize=(18, 12))
    fig1.suptitle('2D Results', fontsize=16)
    
    # 网格对比图
    ax1 = fig1.add_subplot(2, 3, 1)
    ax1.triplot(tri, 'b-', lw=0.8, alpha=0.7, label='original mesh')
    deformed_tri = Triangulation(deformed_nodes[:,0], deformed_nodes[:,1], elements)
    ax1.triplot(deformed_tri, 'r-', lw=0.8, label=f'changed mesh(plus{params.scale_factor}times disp)')
    ax1.set_title('Mesh Comparison')
    ax1.legend()
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # u方向位移
    ax2 = fig1.add_subplot(2, 3, 2)
    contour2 = ax2.tricontourf(tri, u_fem, levels=40, cmap='coolwarm')
    ax2.tricontour(tri, u_fem, levels=10, colors='k', linewidths=0.5, alpha=0.5)
    fig1.colorbar(contour2, ax=ax2, label='disp (m)')
    ax2.set_title('u disp (FEM)')
    ax2.axis('equal')
    
    # v方向位移
    ax3 = fig1.add_subplot(2, 3, 3)
    contour3 = ax3.tricontourf(tri, v_fem, levels=40, cmap='coolwarm')
    ax3.tricontour(tri, v_fem, levels=10, colors='k', linewidths=0.5, alpha=0.5)
    fig1.colorbar(contour3, ax=ax3, label='disp (m)')
    ax3.set_title('v disp (FEM)')
    ax3.axis('equal')
    
    # u方向误差
    ax4 = fig1.add_subplot(2, 3, 4)
    contour4 = ax4.tricontourf(tri, u_error, levels=40, cmap='hot')
    fig1.colorbar(contour4, ax=ax4, label='error (m)')
    ax4.set_title('u error')
    ax4.axis('equal')
    
    # v方向误差
    ax5 = fig1.add_subplot(2, 3, 5)
    contour5 = ax5.tricontourf(tri, v_error, levels=40, cmap='hot')
    fig1.colorbar(contour5, ax=ax5, label='error (m)')
    ax5.set_title('v error')
    ax5.axis('equal')
    
    # 总误差
    ax6 = fig1.add_subplot(2, 3, 6)
    contour6 = ax6.tricontourf(tri, total_error, levels=40, cmap='hot')
    fig1.colorbar(contour6, ax=ax6, label='error (m)')
    ax6.set_title('L2 error (total)')
    ax6.axis('equal')
    
    plt.tight_layout()
    
    # ========== 第二个Figure：3D云图 ==========
    fig2 = plt.figure(figsize=(18, 12))
    fig2.suptitle('3D Displacement Comparison', fontsize=16)
    
    # u方向解析解三维云图
    ax7 = fig2.add_subplot(2, 2, 1, projection='3d')
    surf7 = ax7.plot_trisurf(nodes[:,0], nodes[:,1], u_ana, 
                           triangles=elements, cmap='viridis', edgecolor='none')
    fig2.colorbar(surf7, ax=ax7, shrink=0.5, aspect=10, label='Displacement (m)')
    ax7.set_title('Analytical: u displacement')
    ax7.view_init(elev=30, azim=45)
    
    # u方向有限元解三维云图
    ax8 = fig2.add_subplot(2, 2, 2, projection='3d')
    surf8 = ax8.plot_trisurf(nodes[:,0], nodes[:,1], u_fem, 
                           triangles=elements, cmap='viridis', edgecolor='none')
    fig2.colorbar(surf8, ax=ax8, shrink=0.5, aspect=10, label='Displacement (m)')
    ax8.set_title('FEM: u displacement')
    ax8.view_init(elev=30, azim=45)
    
    # v方向解析解三维云图
    ax9 = fig2.add_subplot(2, 2, 3, projection='3d')
    surf9 = ax9.plot_trisurf(nodes[:,0], nodes[:,1], v_ana, 
                           triangles=elements, cmap='plasma', edgecolor='none')
    fig2.colorbar(surf9, ax=ax9, shrink=0.5, aspect=10, label='Displacement (m)')
    ax9.set_title('Analytical: v displacement')
    ax9.view_init(elev=30, azim=45)
    
    # v方向有限元解三维云图
    ax10 = fig2.add_subplot(2, 2, 4, projection='3d')
    surf10 = ax10.plot_trisurf(nodes[:,0], nodes[:,1], v_fem, 
                            triangles=elements, cmap='plasma', edgecolor='none')
    fig2.colorbar(surf10, ax=ax10, shrink=0.5, aspect=10, label='Displacement (m)')
    ax10.set_title('FEM: v displacement')
    ax10.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    
    # 保存图形
    fig1.savefig('2d_results.png', dpi=300, bbox_inches='tight')
    fig2.savefig('3d_results.png', dpi=300, bbox_inches='tight')
    
    plt.show()

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 初始化参数
    params = BeamParameters()
    
    # 生成网格
    nodes, elements = generate_mesh(params, nx=30, ny=7)
    print(f"网格生成完成: {nodes.shape[0]}节点, {elements.shape[0]}单元")
    
    # 有限元计算
    start_time = time.time()
    U_fem = fem_solution(nodes, elements, params)
    print(f"有限元计算完成: {time.time()-start_time:.2f}秒")
    
    # 计算误差
    u_fem = U_fem[::2]
    v_fem = U_fem[1::2]
    error = np.zeros_like(u_fem)
    for i, (x, y) in enumerate(nodes):
        u_ana, v_ana = analytical_solution(x, y, params)
        error[i] = np.sqrt((u_fem[i]-u_ana)**2 + (v_fem[i]-v_ana)**2)
    print(f"最大位移误差: {np.max(error):.2e} m")
    
    # 可视化所有结果
    plot_results(nodes, elements, U_fem, params)