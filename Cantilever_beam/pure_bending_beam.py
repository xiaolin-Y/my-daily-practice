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
        self.E = 2.1e11        # 弹性模量(Pa) - 修正为钢材典型值
        self.nu = 0.3          # 泊松比
        self.t = 0.1           # 厚度(m) - 更合理的厚度
        self.M = 1000.0        # 自由端弯矩(N·m) - 增大以便观察变形
        self.I = self.t * self.D**3 / 12  # 截面惯性矩(m^4)
        self.scale_factor = 5  # 位移可视化放大系数 - 调整为更合理的值

# ==================== 2. 解析解函数 ====================
def analytical_solution(x, y, params):
    """纯弯情况下的正确解析解（考虑泊松效应）"""
    # 水平位移 (u方向)
    u = params.M * x * y / (params.E * params.I)
    
    # 垂直位移 (v方向)
    v = -params.M * (x**2 + params.nu * y**2) / (2 * params.E * params.I)
    
    return u, v

# ==================== 3. 网格生成 ====================
def generate_mesh(params, nx=30, ny=7):
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
            # 两个三角形组成一个四边形单元
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
    
    # 固定左端节点 (x=0)
    fixed_nodes = [i for i in range(nnodes) if np.isclose(nodes[i,0], 0)]
    fixed_dofs = []
    for n in fixed_nodes:
        # 完全固定 (u=v=0)
        fixed_dofs.extend([2*n, 2*n+1])
    
    # 在右端施加弯矩 (x=L)
    free_end_nodes = [i for i in range(nnodes) if np.isclose(nodes[i,0], params.L)]
    if len(free_end_nodes) >= 2:
        # 找到自由端最上端和最下端的节点
        top_nodes = [n for n in free_end_nodes if nodes[n,1] > 0]
        bottom_nodes = [n for n in free_end_nodes if nodes[n,1] < 0]
        
        if top_nodes and bottom_nodes:
            # 取y值最大和最小的节点
            top_node = max(top_nodes, key=lambda n: nodes[n,1])
            bottom_node = min(bottom_nodes, key=lambda n: nodes[n,1])
            
            # 计算力偶的力大小 (M = F * h)
            h = nodes[top_node,1] - nodes[bottom_node,1]
            F_val = params.M / h
            
            # 施加力偶
            F[2*top_node+1] = F_val    # 上节点 +y 方向力
            F[2*bottom_node+1] = -F_val # 下节点 -y 方向力
    
    # 求解方程组
    free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)
    K_free = K[free_dofs,:][:,free_dofs]
    F_free = F[free_dofs]
    
    # 只求解自由度的位移
    U[free_dofs] = spsolve(K_free, F_free)
    
    return U

def assemble_stiffness_matrix(nodes, elements, params):
    """组装刚度矩阵"""
    nnodes = nodes.shape[0]
    K = lil_matrix((2*nnodes, 2*nnodes))
    
    # 平面应力本构矩阵
    D = params.E/(1-params.nu**2) * np.array([
        [1, params.nu, 0],
        [params.nu, 1, 0],
        [0, 0, (1-params.nu)/2]
    ])
    
    for elem in elements:
        x = nodes[elem, 0]
        y = nodes[elem, 1]
        
        Ke = triangle_stiffness_matrix(x, y, D, params.t)
        
        # 组装到全局刚度矩阵
        dofs = np.array([2*elem[0], 2*elem[0]+1,
                        2*elem[1], 2*elem[1]+1,
                        2*elem[2], 2*elem[2]+1])
        
        for i in range(6):
            for j in range(6):
                K[dofs[i], dofs[j]] += Ke[i,j]
    
    return csr_matrix(K)

def triangle_stiffness_matrix(x, y, D, t):
    """计算三角形单元刚度矩阵"""
    area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (y[1]-y[0])*(x[2]-x[0]))
    
    # 计算几何矩阵B
    b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
    c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
    
    B = np.zeros((3,6))
    B[0,0::2] = b
    B[1,1::2] = c
    B[2,0::2] = c
    B[2,1::2] = b
    B /= (2*area)
    
    # 单元刚度矩阵 Ke = B^T * D * B * t * A
    return t * area * B.T @ D @ B

# ==================== 5. 结果可视化 ====================
def plot_results(nodes, elements, U_fem, params):
    """绘制所有结果图形"""
    plt.close('all')
    
    # 准备数据
    tri = Triangulation(nodes[:,0], nodes[:,1], elements)
    u_fem = U_fem[::2]
    v_fem = U_fem[1::2]
    
    # 计算解析解
    u_ana = np.zeros_like(u_fem)
    v_ana = np.zeros_like(v_fem)
    for i, (x, y) in enumerate(nodes):
        u_ana[i], v_ana[i] = analytical_solution(x, y, params)
    
    # 计算误差
    u_error = u_fem - u_ana
    v_error = v_fem - v_ana
    total_error = np.sqrt(u_error**2 + v_error**2)
    
    # 变形后的节点坐标
    deformed_nodes = nodes + params.scale_factor * np.column_stack((u_fem, v_fem))
    
    # ========== 第一个Figure：2D图 ==========
    fig1 = plt.figure(figsize=(18, 12))
    fig1.suptitle('纯弯悬臂梁分析结果', fontsize=16)
    
    # 网格对比图
    ax1 = fig1.add_subplot(2, 3, 1)
    ax1.triplot(tri, 'b-', lw=0.8, alpha=0.7, label='原始网格')
    deformed_tri = Triangulation(deformed_nodes[:,0], deformed_nodes[:,1], elements)
    ax1.triplot(deformed_tri, 'r-', lw=0.8, alpha=0.7, label=f'变形网格(放大{params.scale_factor}倍)')
    ax1.set_title('网格变形对比')
    ax1.legend()
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # u方向位移
    ax2 = fig1.add_subplot(2, 3, 2)
    contour2 = ax2.tricontourf(tri, u_fem*1e3, levels=20, cmap='coolwarm')  # 转换为mm
    ax2.tricontour(tri, u_fem*1e3, levels=10, colors='k', linewidths=0.5, alpha=0.5)
    fig1.colorbar(contour2, ax=ax2, label='位移 (mm)')
    ax2.set_title('u方向位移 (FEM)')
    ax2.axis('equal')
    
    # v方向位移
    ax3 = fig1.add_subplot(2, 3, 3)
    contour3 = ax3.tricontourf(tri, v_fem*1e3, levels=20, cmap='coolwarm')  # 转换为mm
    ax3.tricontour(tri, v_fem*1e3, levels=10, colors='k', linewidths=0.5, alpha=0.5)
    fig1.colorbar(contour3, ax=ax3, label='位移 (mm)')
    ax3.set_title('v方向位移 (FEM)')
    ax3.axis('equal')
    
    # u方向误差
    ax4 = fig1.add_subplot(2, 3, 4)
    contour4 = ax4.tricontourf(tri, u_error*1e6, levels=20, cmap='hot')  # 转换为μm
    fig1.colorbar(contour4, ax=ax4, label='误差 (μm)')
    ax4.set_title('u方向误差 (FEM - 解析解)')
    ax4.axis('equal')
    
    # v方向误差
    ax5 = fig1.add_subplot(2, 3, 5)
    contour5 = ax5.tricontourf(tri, v_error*1e6, levels=20, cmap='hot')  # 转换为μm
    fig1.colorbar(contour5, ax=ax5, label='误差 (μm)')
    ax5.set_title('v方向误差 (FEM - 解析解)')
    ax5.axis('equal')
    
    # 总误差
    ax6 = fig1.add_subplot(2, 3, 6)
    contour6 = ax6.tricontourf(tri, total_error*1e6, levels=20, cmap='hot')  # 转换为μm
    fig1.colorbar(contour6, ax=ax6, label='误差 (μm)')
    ax6.set_title('总位移误差 (FEM - 解析解)')
    ax6.axis('equal')
    
    plt.tight_layout()
    
    # ========== 第二个Figure：3D云图 ==========
    fig2 = plt.figure(figsize=(18, 12))
    fig2.suptitle('纯弯悬臂梁3D位移对比', fontsize=16)
    
    # u方向解析解三维云图
    ax7 = fig2.add_subplot(2, 2, 1, projection='3d')
    surf7 = ax7.plot_trisurf(nodes[:,0], nodes[:,1], u_ana*1e3, 
                           triangles=elements, cmap='viridis', edgecolor='none')
    fig2.colorbar(surf7, ax=ax7, shrink=0.5, aspect=10, label='位移 (mm)')
    ax7.set_title('解析解: u方向位移')
    ax7.view_init(elev=30, azim=45)
    ax7.set_zlabel('位移 (mm)')
    
    # u方向有限元解三维云图
    ax8 = fig2.add_subplot(2, 2, 2, projection='3d')
    surf8 = ax8.plot_trisurf(nodes[:,0], nodes[:,1], u_fem*1e3, 
                           triangles=elements, cmap='viridis', edgecolor='none')
    fig2.colorbar(surf8, ax=ax8, shrink=0.5, aspect=10, label='位移 (mm)')
    ax8.set_title('有限元解: u方向位移')
    ax8.view_init(elev=30, azim=45)
    ax8.set_zlabel('位移 (mm)')
    
    # v方向解析解三维云图
    ax9 = fig2.add_subplot(2, 2, 3, projection='3d')
    surf9 = ax9.plot_trisurf(nodes[:,0], nodes[:,1], v_ana*1e3, 
                           triangles=elements, cmap='plasma', edgecolor='none')
    fig2.colorbar(surf9, ax=ax9, shrink=0.5, aspect=10, label='位移 (mm)')
    ax9.set_title('解析解: v方向位移')
    ax9.view_init(elev=30, azim=45)
    ax9.set_zlabel('位移 (mm)')
    
    # v方向有限元解三维云图
    ax10 = fig2.add_subplot(2, 2, 4, projection='3d')
    surf10 = ax10.plot_trisurf(nodes[:,0], nodes[:,1], v_fem*1e3, 
                            triangles=elements, cmap='plasma', edgecolor='none')
    fig2.colorbar(surf10, ax=ax10, shrink=0.5, aspect=10, label='位移 (mm)')
    ax10.set_title('有限元解: v方向位移')
    ax10.view_init(elev=30, azim=45)
    ax10.set_zlabel('位移 (mm)')
    
    plt.tight_layout()
    
    # 保存图形
    fig1.savefig('beam_bending_2d_results.png', dpi=300, bbox_inches='tight')
    fig2.savefig('beam_bending_3d_results.png', dpi=300, bbox_inches='tight')
    
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
    max_error = 0
    for i, (x, y) in enumerate(nodes):
        u_ana, v_ana = analytical_solution(x, y, params)
        error = np.sqrt((u_fem[i]-u_ana)**2 + (v_fem[i]-v_ana)**2)
        if error > max_error:
            max_error = error
    print(f"最大位移误差: {max_error:.2e} m")
    
    # 可视化结果
    plot_results(nodes, elements, U_fem, params)