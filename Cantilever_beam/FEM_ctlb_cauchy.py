import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


# ==================== 1. 参数定义 ====================
class BeamParameters:
    def __init__(self):
        self.L = 10.0          # 梁长度(m)
        self.D = 2.0           # 梁高度(m)
        self.E = 2.1e5         # 弹性模量(Pa)
        self.nu = 0.3          # 泊松比
        self.t = 1.0           # 厚度(m)
        self.total_force = -300.0  # 总力大小(N)
        self.I = self.t * self.D**3 / 12  # 截面惯性矩(m^4)
        self.scale_factor = 5  # 位移可视化放大系数

# ==================== 2. 解析解函数 ====================
def analytical_solution(x, y, params):
    """Timoshenko梁理论解析解"""
    term1 = (6*params.L - 3*x)*x
    term2 = (2 + params.nu)*(y**2 - params.D**2/4)
    u = -params.total_force*y/(6*params.E*params.I) * (term1 + term2)
    
    term3 = 3*params.nu*y**2*(params.L - x)
    term4 = (4 + 5*params.nu)*params.D**2*x/4
    term5 = (3*params.L - x)*x**2
    v = params.total_force/(6*params.E*params.I) * (term3 + term4 + term5)
    return u, v

def analytical_stress(x, y, params):
    """计算解析解的应力分量"""
    sigma_x = -params.total_force * (params.L - x) * y / params.I
    tau_xy = params.total_force / (2 * params.I) * ((params.D ** 2) / 4 - y ** 2)
    sigma_y = 0.0
    return sigma_x, tau_xy, sigma_y

# ==================== 3. 网格生成 ====================
def generate_mesh(params, nx, ny):
    """生成结构化四边形网格"""
    x = np.linspace(0, params.L, nx)
    y = np.linspace(-params.D/2, params.D/2, ny)
    X, Y = np.meshgrid(x, y)
    nodes = np.column_stack((X.flatten(), Y.flatten()))
    
    elements = []
    for j in range(ny-1):
        for i in range(nx-1):
            # 四边形单元节点索引（顺时针顺序）
            n1 = j*nx + i
            n2 = j*nx + i + 1
            n3 = (j + 1)*nx + i + 1
            n4 = (j + 1)*nx + i
            elements.append([n1, n2, n3, n4])
    
    return nodes, np.array(elements), nx, ny  # 返回nx和ny用于后续可视化

# ==================== 4. 有限元计算 ====================
def fem_solution(nodes, elements, params):
    """执行有限元计算"""
    nnodes = nodes.shape[0]
    ndof = 2 * nnodes  # 每个节点2个自由度
    
    # 组装刚度矩阵
    K = assemble_stiffness_matrix(nodes, elements, params)
    
    # 初始化位移和力向量
    U = np.zeros(ndof)
    F = np.zeros(ndof)
    
    # 固定端位移约束 (x=0处)
    fixed_nodes = [i for i in range(nnodes) if np.isclose(nodes[i,0], 0)]
    fixed_dofs = []
    for n in fixed_nodes:
        u, v = analytical_solution(nodes[n,0], nodes[n,1], params)
        U[2*n] = u
        U[2*n+1] = v
        fixed_dofs.extend([2*n, 2*n+1])
    
    # 应用柯西公式的分布面力
    apply_cauchy_distributed_force(nodes, elements, F, params)
    
    # 求解
    free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)
    K_free = K[free_dofs,:][:,free_dofs]
    F_eff = F[free_dofs] - K[free_dofs,:][:,fixed_dofs] @ U[fixed_dofs]
    
    U[free_dofs] = spsolve(K_free, F_eff)
    
    return U
    
        
def apply_cauchy_distributed_force(nodes, elements, F, params):
    """基于柯西公式在梁自由端施加分布面力"""
    # 确定自由端节点（x = L）
    free_end_nodes = [i for i in range(nodes.shape[0]) if np.isclose(nodes[i,0], params.L)]
    
    if not free_end_nodes:
        raise ValueError("未找到自由端节点")
    
    # 法向量（沿x轴正方向）
    n = np.array([1.0, 0.0])  

    for node_idx in free_end_nodes:
        x = params.L  # 自由端 x 坐标
        y = nodes[node_idx, 1]  # 节点 y 坐标
        
        # 计算应力分量
        sigma_x = -params.total_force * (params.L - x) * y / params.I
        tau_xy = params.total_force / (2 * params.I) * ((params.D ** 2) / 4 - y ** 2)
        sigma_y = 0.0
        
        # 构造应力张量
        stress_tensor = np.array([[sigma_x, tau_xy],
                                  [tau_xy, sigma_y]])
        
        # 应用柯西公式计算面力：t = σ · n
        surface_force = np.dot(stress_tensor, n)
        
        # 计算节点贡献的力
        dy = (params.D) / (len(free_end_nodes) - 1) if len(free_end_nodes) > 1 else params.D
        force_contribution = surface_force * params.t * dy
        
        # 应用到力向量
        F[2 * node_idx] += force_contribution[0]    # x 方向力
        F[2 * node_idx + 1] += force_contribution[1]  # y 方向力
    

def assemble_stiffness_matrix(nodes, elements, params):
    """组装刚度矩阵"""
    nnodes = nodes.shape[0]
    K = lil_matrix((2*nnodes, 2*nnodes))
    
    # 本构矩阵 (平面应力)
    D = params.E/(1-params.nu**2) * np.array([
        [1, params.nu, 0],
        [params.nu, 1, 0],
        [0, 0, (1-params.nu)/2]
    ])
    
    for elem in elements:
        # 获取单元节点坐标
        coords = nodes[elem, :]
        x = coords[:, 0]
        y = coords[:, 1]
        
        # 计算四边形单元刚度矩阵
        Ke = quadrilateral_stiffness_matrix(x, y, D, params.t)
        
        # 单元自由度索引
        dofs = np.array([2*elem[0], 2*elem[0]+1,
                        2*elem[1], 2*elem[1]+1,
                        2*elem[2], 2*elem[2]+1,
                        2*elem[3], 2*elem[3]+1]).flatten()
        
        # 组装到整体刚度矩阵
        for i in range(8):
            for j in range(8):
                K[dofs[i], dofs[j]] += Ke[i, j]
    
    return csr_matrix(K)

def quadrilateral_stiffness_matrix(x, y, D, t):
    """计算四边形单元刚度矩阵（采用4节点等参元）"""
    Ke = np.zeros((8, 8))
    # 高斯积分点 (2x2积分)
    gauss_pts = np.array([[-1/np.sqrt(3), -1/np.sqrt(3)],
                         [1/np.sqrt(3), -1/np.sqrt(3)],
                         [1/np.sqrt(3), 1/np.sqrt(3)],
                         [-1/np.sqrt(3), 1/np.sqrt(3)]])
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    
    for i, (xi, eta) in enumerate(gauss_pts):
        # 形状函数导数 (对ξ, η)
        dN_dxi = 0.25 * np.array([
            [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
            [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]
        ])
        
        # 计算雅可比矩阵
        J = np.zeros((2, 2))
        for j in range(4):
            J[0, 0] += dN_dxi[0, j] * x[j]
            J[0, 1] += dN_dxi[0, j] * y[j]
            J[1, 0] += dN_dxi[1, j] * x[j]
            J[1, 1] += dN_dxi[1, j] * y[j]
        
        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)
        
        # B矩阵 (应变-位移矩阵)
        B = np.zeros((3, 8))
        dN_dx = invJ @ dN_dxi
        
        for j in range(4):
            B[0, 2*j] = dN_dx[0, j]  # du/dx
            B[1, 2*j+1] = dN_dx[1, j]  # dv/dy
            B[2, 2*j] = dN_dx[1, j]  # du/dy
            B[2, 2*j+1] = dN_dx[0, j]  # dv/dx
        
        # 累加刚度矩阵
        Ke += weights[i] * detJ * B.T @ D @ B
    
    return Ke * t

# 从有限元位移计算应力
def fem_stress(nodes, elements, U_fem, params):
    """从有限元位移结果计算应力"""
    nnodes = nodes.shape[0]
    stress_sigma_x = np.zeros(nnodes)
    stress_tau_xy = np.zeros(nnodes)
    count = np.zeros(nnodes)  # 用于平均
    
    # 本构矩阵
    D = params.E/(1-params.nu**2) * np.array([
        [1, params.nu, 0],
        [params.nu, 1, 0],
        [0, 0, (1-params.nu)/2]
    ])
    
    for elem in elements:
        # 获取单元节点坐标和位移
        coords = nodes[elem, :]
        x = coords[:, 0]
        y = coords[:, 1]
        u = np.array([U_fem[2*elem[0]], U_fem[2*elem[0]+1],
                     U_fem[2*elem[1]], U_fem[2*elem[1]+1],
                     U_fem[2*elem[2]], U_fem[2*elem[2]+1],
                     U_fem[2*elem[3]], U_fem[2*elem[3]+1]])
        
        # 高斯积分点
        gauss_pts = np.array([[0, 0]])  # 单元中心积分点
        
        for (xi, eta) in gauss_pts:
            # 形状函数导数
            dN_dxi = 0.25 * np.array([
                [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]
            ])
            
            # 雅可比矩阵
            J = np.zeros((2, 2))
            for j in range(4):
                J[0, 0] += dN_dxi[0, j] * x[j]
                J[0, 1] += dN_dxi[0, j] * y[j]
                J[1, 0] += dN_dxi[1, j] * x[j]
                J[1, 1] += dN_dxi[1, j] * y[j]
            
            invJ = np.linalg.inv(J)
            dN_dx = invJ @ dN_dxi
            
            # B矩阵
            B = np.zeros((3, 8))
            for j in range(4):
                B[0, 2*j] = dN_dx[0, j]
                B[1, 2*j+1] = dN_dx[1, j]
                B[2, 2*j] = dN_dx[1, j]
                B[2, 2*j+1] = dN_dx[0, j]
            
            # 计算应变和应力
            strain = B @ u
            stress = D @ strain
            
            # 应力分配到节点
            for j, idx in enumerate(elem):
                stress_sigma_x[idx] += stress[0]
                stress_tau_xy[idx] += stress[2]
                count[idx] += 1
    
    # 平均节点应力
    stress_sigma_x /= count
    stress_tau_xy /= count
    
    return stress_sigma_x, stress_tau_xy

# ==================== 5. 结果可视化 ====================
def plot_results(nodes, elements, U_fem, params, nx, ny):
    """绘制所有结果图形，使用pcolormesh显示连续云图"""
    plt.close('all')
    
    # 准备数据
    u_fem = U_fem[::2]
    v_fem = U_fem[1::2]
    
    u_ana = np.zeros_like(u_fem)
    v_ana = np.zeros_like(v_fem)
    sigma_x_ana = np.zeros_like(u_fem)
    tau_xy_ana = np.zeros_like(u_fem)
    
    for i, (x, y) in enumerate(nodes):
        u_ana[i], v_ana[i] = analytical_solution(x, y, params)
        sigma_x_ana[i], tau_xy_ana[i], _ = analytical_stress(x, y, params)
    
    # 计算有限元应力
    sigma_x_fem, tau_xy_fem = fem_stress(nodes, elements, U_fem, params)
    
    # 计算误差
    sigma_x_error = np.abs(sigma_x_fem - sigma_x_ana)
    tau_xy_error = np.abs(tau_xy_fem - tau_xy_ana)
    u_error = np.abs(u_fem - u_ana)
    v_error = np.abs(v_fem - v_ana)
    total_error = np.sqrt(u_error**2 + v_error**2)
    deformed_nodes = nodes + params.scale_factor * np.column_stack((u_fem, v_fem))
    
    # 重塑为网格形状用于pcolormesh
    x_grid = nodes[:,0].reshape(ny, nx)
    y_grid = nodes[:,1].reshape(ny, nx)
    u_grid = u_fem.reshape(ny, nx)
    v_grid = v_fem.reshape(ny, nx)
    u_error_grid = u_error.reshape(ny, nx)
    v_error_grid = v_error.reshape(ny, nx)
    total_error_grid = total_error.reshape(ny, nx)
    sigma_x_fem_grid = sigma_x_fem.reshape(ny, nx)
    tau_xy_fem_grid = tau_xy_fem.reshape(ny, nx)
    sigma_x_ana_grid = sigma_x_ana.reshape(ny, nx)
    tau_xy_ana_grid = tau_xy_ana.reshape(ny, nx)
    u_ana_grid = u_ana.reshape(ny, nx)
    v_ana_grid = v_ana.reshape(ny, nx)
    
    # ========== 第一个Figure：2D图 ==========
    fig1 = plt.figure(figsize=(18, 12))
    fig1.suptitle('2D Results', fontsize=16)
    
    # 网格对比图
    ax1 = fig1.add_subplot(2, 3, 1)

    # 绘制原始网格并保存绘图对象
    original_mesh = []
    for elem in elements:
        polygon = patches.Polygon(nodes[elem], edgecolor='blue', facecolor='none', 
                             linewidth=0.8, alpha=0.7)
        ax1.add_patch(polygon)
        original_mesh.append(polygon)

    # 绘制变形网格并保存绘图对象
    deformed_mesh = []
    for elem in elements:
        polygon = patches.Polygon(deformed_nodes[elem], edgecolor='red', facecolor='none', 
                             linewidth=0.8, alpha=0.7)
        ax1.add_patch(polygon)
        deformed_mesh.append(polygon)

    # 使用绘图对象的第一个元素作为图例句柄，确保颜色匹配
    ax1.legend(
        [original_mesh[0], deformed_mesh[0]],  # 句柄
        ['Original mesh', f'Deformed mesh (x{params.scale_factor})'],  # 标签
        loc='best'
)
    ax1.set_title('Mesh Comparison')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # u方向位移 - 使用pcolormesh
    ax2 = fig1.add_subplot(2, 3, 2)
    contour2 = ax2.pcolormesh(x_grid, y_grid, u_grid, cmap='coolwarm', shading='gouraud')
    fig1.colorbar(contour2, ax=ax2, label='Displacement (m)')
    ax2.set_title('u displacement (FEM)')
    ax2.axis('equal')
    
    # v方向位移 - 使用pcolormesh
    ax3 = fig1.add_subplot(2, 3, 3)
    contour3 = ax3.pcolormesh(x_grid, y_grid, v_grid, cmap='coolwarm', shading='gouraud')
    fig1.colorbar(contour3, ax=ax3, label='Displacement (m)')
    ax3.set_title('v displacement (FEM)')
    ax3.axis('equal')
    
    # u方向误差 - 使用pcolormesh
    ax4 = fig1.add_subplot(2, 3, 4)
    contour4 = ax4.pcolormesh(x_grid, y_grid, u_error_grid, cmap='hot', shading='gouraud')
    fig1.colorbar(contour4, ax=ax4, label='Error (m)')
    ax4.set_title('u displacement error')
    ax4.axis('equal')
    
    # v方向误差 - 使用pcolormesh
    ax5 = fig1.add_subplot(2, 3, 5)
    contour5 = ax5.pcolormesh(x_grid, y_grid, v_error_grid, cmap='hot', shading='gouraud')
    fig1.colorbar(contour5, ax=ax5, label='Error (m)')
    ax5.set_title('v displacement error')
    ax5.axis('equal')
    
    # 总误差 - 使用pcolormesh
    ax6 = fig1.add_subplot(2, 3, 6)
    contour6 = ax6.pcolormesh(x_grid, y_grid, total_error_grid, cmap='hot', shading='gouraud')
    fig1.colorbar(contour6, ax=ax6, label='Error (m)')
    ax6.set_title('Total displacement error')
    ax6.axis('equal')
    
    plt.tight_layout()
    
    # ========== 第二个Figure：3D云图（位移） ==========
    fig2 = plt.figure(figsize=(18, 12))
    fig2.suptitle('3D Displacement Comparison', fontsize=16)
    
    # u方向解析解三维云图
    ax7 = fig2.add_subplot(2, 2, 1, projection='3d')
    surf7 = ax7.plot_surface(x_grid, y_grid, u_ana_grid, cmap='viridis', edgecolor='none')
    fig2.colorbar(surf7, ax=ax7, shrink=0.5, aspect=10, label='Displacement (m)')
    ax7.set_title('Analytical: u displacement')
    ax7.view_init(elev=30, azim=45)
    
    # u方向有限元解三维云图
    ax8 = fig2.add_subplot(2, 2, 2, projection='3d')
    surf8 = ax8.plot_surface(x_grid, y_grid, u_grid, cmap='viridis', edgecolor='none')
    fig2.colorbar(surf8, ax=ax8, shrink=0.5, aspect=10, label='Displacement (m)')
    ax8.set_title('FEM: u displacement')
    ax8.view_init(elev=30, azim=45)
    
    # v方向解析解三维云图
    ax9 = fig2.add_subplot(2, 2, 3, projection='3d')
    surf9 = ax9.plot_surface(x_grid, y_grid, v_ana_grid, cmap='plasma', edgecolor='none')
    fig2.colorbar(surf9, ax=ax9, shrink=0.5, aspect=10, label='Displacement (m)')
    ax9.set_title('Analytical: v displacement')
    ax9.view_init(elev=30, azim=45)
    
    # v方向有限元解三维云图
    ax10 = fig2.add_subplot(2, 2, 4, projection='3d')
    surf10 = ax10.plot_surface(x_grid, y_grid, v_grid, cmap='plasma', edgecolor='none')
    fig2.colorbar(surf10, ax=ax10, shrink=0.5, aspect=10, label='Displacement (m)')
    ax10.set_title('FEM: v displacement')
    ax10.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    
    # ========== 第三个Figure：3D云图（应力）- 放大版本 ==========
    fig3 = plt.figure(figsize=(24, 18))  # 显著增大图的尺寸
    fig3.suptitle('3D Stress Comparison', fontsize=20, y=0.98)  # 调整标题位置

    # 正应力sigma_x解析解
    ax11 = fig3.add_subplot(2, 2, 1, projection='3d')
    surf11 = ax11.plot_surface(x_grid, y_grid, sigma_x_ana_grid, cmap='inferno', edgecolor='none')
    fig3.colorbar(surf11, ax=ax11, shrink=0.7, aspect=15, label='Stress (Pa)')  # 调整色条大小
    ax11.set_title('Analytical: σ_x stress', fontsize=16)
    ax11.set_xlabel('X (m)', fontsize=12)
    ax11.set_ylabel('Y (m)', fontsize=12)
    ax11.set_zlabel('Stress (Pa)', fontsize=12)
    ax11.tick_params(axis='both', which='major', labelsize=10)
    ax11.view_init(elev=30, azim=45)

    # 正应力sigma_x有限元解
    ax12 = fig3.add_subplot(2, 2, 2, projection='3d')
    surf12 = ax12.plot_surface(x_grid, y_grid, sigma_x_fem_grid-sigma_x_ana_grid, cmap='inferno', edgecolor='none')
    fig3.colorbar(surf12, ax=ax12, shrink=0.7, aspect=15, label='Stress (Pa)')
    ax12.set_title('σ_x stress error', fontsize=16)
    ax12.set_xlabel('X (m)', fontsize=12)
    ax12.set_ylabel('Y (m)', fontsize=12)
    ax12.set_zlabel('Stress (Pa)', fontsize=12)
    ax12.tick_params(axis='both', which='major', labelsize=10)
    ax12.view_init(elev=30, azim=45)

    # 切应力tau_xy解析解
    ax13 = fig3.add_subplot(2, 2, 3, projection='3d')
    surf13 = ax13.plot_surface(x_grid, y_grid, tau_xy_ana_grid, cmap='magma', edgecolor='none')
    fig3.colorbar(surf13, ax=ax13, shrink=0.7, aspect=15, label='Stress (Pa)')
    ax13.set_title('Analytical: τ_xy stress', fontsize=16)
    ax13.set_xlabel('X (m)', fontsize=12)
    ax13.set_ylabel('Y (m)', fontsize=12)
    ax13.set_zlabel('Stress (Pa)', fontsize=12)
    ax13.tick_params(axis='both', which='major', labelsize=10)
    ax13.view_init(elev=30, azim=45)

    # 切应力tau_xy有限元解
    ax14 = fig3.add_subplot(2, 2, 4, projection='3d')
    surf14 = ax14.plot_surface(x_grid, y_grid, tau_xy_fem_grid - tau_xy_ana_grid, cmap='magma', edgecolor='none')
    fig3.colorbar(surf14, ax=ax14, shrink=0.7, aspect=15, label='Stress (Pa)')
    ax14.set_title('τ_xy stress error', fontsize=16)
    ax14.set_xlabel('X (m)', fontsize=12)
    ax14.set_ylabel('Y (m)', fontsize=12)
    ax14.set_zlabel('Stress (Pa)', fontsize=12)
    ax14.tick_params(axis='both', which='major', labelsize=10)
    ax14.view_init(elev=30, azim=45)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，为标题留出空间
    
    # 保存图形
    fig1.savefig('2d_results.png', dpi=300, bbox_inches='tight')
    fig2.savefig('3d_displacement_results.png', dpi=300, bbox_inches='tight')
    fig3.savefig('3d_stress_results.png', dpi=300, bbox_inches='tight')
    
    plt.show()