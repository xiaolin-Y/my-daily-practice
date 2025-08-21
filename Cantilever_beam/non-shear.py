import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import time

class FEMCantileverBeamSolver:
    def __init__(self, L=10.0, h=1.0, nx=50, ny=5, E=2.1e5, nu=0.3, F=-1, ngp=2):
        # 几何和材料参数
        self.L = L          # 梁长度(m)
        self.h = h          # 梁高度(m)
        self.nx = nx        # 长度方向单元数
        self.ny = ny        # 高度方向单元数
        self.E = E          # 弹性模量(Pa)
        self.nu = nu        # 泊松比
        self.applied_force = F  # 末端集中力(N)，避免与载荷向量冲突
        self.ngp = ngp      # 积分点数
        
        # 计算派生参数
        self.nnx = nx + 1   # 长度方向节点数
        self.nny = ny + 1   # 高度方向节点数
        self.nnp = self.nnx * self.nny  # 总节点数
        self.nel = nx * ny  # 总单元数
        self.ndof = 2 * self.nnp  # 总自由度

    def generate_mesh(self):
        """生成规则矩形网格（4节点四边形单元）"""
        # 节点坐标 (y方向在前，x方向在后)
        x_coords = np.linspace(0, self.L, self.nnx)
        y_coords = np.linspace(-self.h/2, self.h/2, self.nny)
        X, Y = np.meshgrid(x_coords, y_coords)  # 形状为(nny, nnx)
        self.x = X.flatten() # 将一个多维数组X展平成一维数组，并将结果赋值给类的实例变量self.x
        self.y = Y.flatten()
        
        # 单元连接关系（逆时针编号：左下、右下、右上、左上）
        self.elements = np.zeros((self.nel, 4), dtype=int)
        for j in range(self.ny):  # y方向单元索引
            for i in range(self.nx):  # x方向单元索引
                elem_idx = j * self.nx + i  # 单元编号
                n1 = j * self.nnx + i# 左下节点索引
                self.elements[elem_idx] = [
                    n1,                  # 左下 (i,j)
                    n1 + 1,              # 右下 (i+1,j)
                    n1 + self.nnx + 1,   # 右上 (i+1,j+1)
                    n1 + self.nnx        # 左上 (i,j+1)
                ]

    def assemble_system(self):
        """组装刚度矩阵和载荷向量"""
        # 材料矩阵 (平面应力条件)
        self.D = (self.E / (1 - self.nu**2)) * np.array([
            [1, self.nu, 0],
            [self.nu, 1, 0],
            [0, 0, (1 - self.nu)/2]
        ])
        
        # 初始化刚度矩阵和载荷向量
        self.K = lil_matrix((self.ndof, self.ndof))  # 稀疏矩阵节省内存
        self.R = np.zeros(self.ndof)  # 一维载荷向量（避免与applied_force冲突）
        
        # 高斯积分点和权重（[-1,1]区间）
        gp, gw = np.polynomial.legendre.leggauss(self.ngp)
        
        # 遍历所有单元
        for e in range(self.nel):
            nodes = self.elements[e]  # 单元的四个节点
            coords = np.column_stack((self.x[nodes], self.y[nodes]))  # 4x2坐标矩阵
            Ke = np.zeros((8, 8))  # 单元刚度矩阵 (4节点×2自由度)
            
            # 高斯积分（二维）
            for i in range(self.ngp):
                for j in range(self.ngp):
                    xi, eta = gp[i], gp[j]  # 积分点坐标
                    weight = gw[i] * gw[j]  # 权重乘积
                    
                    # 形函数及其对自然坐标的导数
                    N = self.shape_functions(xi, eta)  # 4x1形函数
                    dN_dxi_eta = self.shape_derivatives(xi, eta)  # 2x4导数矩阵 (dN/dxi, dN/deta)
                    
                    # 雅可比矩阵 (2x2)
                    J = dN_dxi_eta @ coords  # 导数矩阵 × 坐标矩阵
                    detJ = np.linalg.det(J)  # 雅可比行列式
                    if detJ <= 0:
                        raise ValueError(f"单元 {e} 雅可比行列式非正 ({detJ})，节点编号可能错误")
                    invJ = np.linalg.inv(J)  # 雅可比逆矩阵
                    
                    # 形函数对物理坐标的导数 (2x4)
                    dN_dx_dy = invJ.T @ dN_dxi_eta  # (dN/dx, dN/dy)
                    
                    # 应变-位移矩阵B (3x8)
                    B = np.zeros((3, 8))
                    for n in range(4):  # 四个节点
                        B[0, 2*n] = dN_dx_dy[0, n]  # ε_xx = du/dx
                        B[1, 2*n + 1] = dN_dx_dy[1, n]  # ε_yy = dv/dy
                        B[2, 2*n] = dN_dx_dy[1, n]  # γ_xy = du/dy + dv/dx
                        B[2, 2*n + 1] = dN_dx_dy[0, n]
                    
                    # 单元刚度矩阵贡献 (积分)
                    Ke += B.T @ self.D @ B * detJ * weight
            
            # 组装到全局刚度矩阵
            dofs = self.get_dof_indices(nodes)  # 单元自由度索引
            for i in range(8):
                for j in range(8):
                    self.K[dofs[i], dofs[j]] += Ke[i, j]
        
        # 转换为CSR格式以提高求解效率
        self.K = self.K.tocsr()
        
        # 施加末端集中力（平均分配到右端所有节点的y方向）
        right_nodes = np.where(np.isclose(self.x, self.L))[0]  # 右端节点（浮点数安全比较）
        if not len(right_nodes):
            raise ValueError("未找到右端节点，可能网格生成错误")
        force_per_node = self.applied_force / len(right_nodes)
        for node in right_nodes:
            self.R[2*node + 1] += force_per_node  # y方向自由度索引

    def solve(self):
        """求解系统方程（考虑边界条件）"""
        # 边界条件: 固定左端 (x=0处所有节点的x和y方向位移)
        left_nodes = np.where(np.isclose(self.x, 0))[0]  # 左端节点
        fixed_dofs = []
        for node in left_nodes:
            fixed_dofs.extend([2*node, 2*node + 1])  # x和y方向自由度
        fixed_dofs = np.unique(fixed_dofs)  # 去重
        
        # 自由自由度
        all_dofs = np.arange(self.ndof)
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
        
        # 提取缩减系统并求解
        K_ff = self.K[free_dofs, :][:, free_dofs]  # 自由-自由刚度矩阵
        R_f = self.R[free_dofs]  # 自由度载荷
        
        # 求解线性方程组
        u_free = spsolve(K_ff, R_f)
        
        # 组装全位移向量
        self.u = np.zeros(self.ndof)
        self.u[free_dofs] = u_free
        self.u[fixed_dofs] = 0.0  # 固定端位移为0
        
        # 提取节点位移（x和y方向）
        self.u_x = self.u[::2]  # x方向位移 (每隔一个取)
        self.u_y = self.u[1::2]  # y方向位移

    def shape_functions(self, xi, eta):
        """4节点四边形单元形函数（自然坐标）"""
        return 0.25 * np.array([
            (1 - xi) * (1 - eta),  # 节点1
            (1 + xi) * (1 - eta),  # 节点2
            (1 + xi) * (1 + eta),  # 节点3
            (1 - xi) * (1 + eta)   # 节点4
        ])

    def shape_derivatives(self, xi, eta):
        """形函数对自然坐标(xi, eta)的导数"""
        return 0.25 * np.array([
            # 第一行：dN/dxi
            [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
            # 第二行：dN/deta
            [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]
        ])

    def get_dof_indices(self, nodes):
        """获取单元节点对应的全局自由度索引（x方向先，y方向后）"""
        dofs = np.zeros(2*len(nodes), dtype=int)
        for i, node in enumerate(nodes):
            dofs[2*i] = 2*node      # x方向自由度
            dofs[2*i + 1] = 2*node + 1  # y方向自由度
        return dofs

    def analytical_solution(self):
        """基于欧拉-伯努利梁理论的解析解"""
        I = self.h**3 / 12  # 截面惯性矩
        # 计算每个节点的解析位移
        u_x_analy = np.zeros_like(self.x)
        u_y_analy = np.zeros_like(self.x)
        
        for i in range(self.nnp):
            x, y = self.x[i], self.y[i]
            if np.isclose(x, 0):
                u_y_analy[i] = 0.0
                u_x_analy[i] = 0.0
            else:
                # y方向挠度（向下为正）
                u_y_analy[i] = self.applied_force * x**2 * (3*self.L - x) / (6 * self.E * I)
                # x方向位移（轴向）
                u_x_analy[i] = -y * self.applied_force * x * (self.L - x/2) / (self.E * I)
        return u_x_analy, u_y_analy

    def plot_results(self):
        """可视化有限元解、解析解及误差"""
        # 解析解
        u_x_analy, u_y_analy = self.analytical_solution()
        
        # 网格重塑（用于绘图）
        X = self.x.reshape(self.nny, self.nnx)
        Y = self.y.reshape(self.nny, self.nnx)
        u_y_fe = self.u_y.reshape(self.nny, self.nnx)
        u_y_ana = u_y_analy.reshape(self.nny, self.nnx)
        
        # 绘图设置
        plt.figure(figsize=(16, 12))
        plt.suptitle("analysis results of cantilever beam")
        
        # 1. 有限元解 - y方向位移
        plt.subplot(2, 2, 1)
        cf = plt.contourf(X, Y, u_y_fe, 20, cmap='viridis')
        plt.colorbar(cf, label='u/m')
        plt.plot(X, Y, 'k-', linewidth=0.5, alpha=0.3)  # 网格线
        plt.plot(X.T, Y.T, 'k-', linewidth=0.5, alpha=0.3)
        plt.title("fem-y displacement")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.axis('equal') #设置坐标轴的比例相等，确保 x 轴和 y 轴的单位长度相同
        
        # 2. 解析解 - y方向位移
        plt.subplot(2, 2, 2)
        cf = plt.contourf(X, Y, u_y_ana, 20, cmap='viridis')
        plt.colorbar(cf, label='u/m')
        plt.plot(X, Y, 'k-', linewidth=0.5, alpha=0.3)
        plt.plot(X.T, Y.T, 'k-', linewidth=0.5, alpha=0.3)
        plt.title("exact-y displacement")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.axis('equal')
        
        # 3. 位移误差
        error = u_y_fe - u_y_ana
        plt.subplot(2, 2, 3)
        cf = plt.contourf(X, Y, error, 20, cmap='bwr')
        plt.colorbar(cf, label='error/m')
        plt.title(f"error-y (max: {np.max(np.abs(error)):.2e} m)")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.axis('equal')
        
        # 4. 变形网格
        plt.subplot(2, 2, 4)
        scale = 30  # 位移放大系数
        # 原始网格
        plt.plot(X, Y, 'k-', linewidth=0.7, alpha=0.5)
        plt.plot(X.T, Y.T, 'k-', linewidth=0.7, alpha=0.5)
        # 变形后网格
        X_def = X + scale * self.u_x.reshape(self.nny, self.nnx)
        Y_def = Y + scale * self.u_y.reshape(self.nny, self.nnx)
        plt.plot(X_def, Y_def, 'r-', linewidth=0.7, alpha=0.8)
        plt.plot(X_def.T, Y_def.T, 'r-', linewidth=0.7, alpha=0.8)
        plt.title("compared deformed mesh")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.axis('equal')
        plt.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 预留suptitle空间
        
        # 计算末端挠度误差
        tip_nodes = np.where(np.isclose(self.x, self.L))[0]
        fe_tip_deflect = np.mean(self.u_y[tip_nodes])
        ana_tip_deflect = np.mean(u_y_analy[tip_nodes])
        rel_error = np.abs(fe_tip_deflect - ana_tip_deflect) / np.abs(ana_tip_deflect) * 100
        
        print(f"\n末端挠度对比:")
        print(f"  有限元解: {fe_tip_deflect:.6f} m")
        print(f"  解析解:   {ana_tip_deflect:.6f} m")
        print(f"  相对误差: {rel_error:.2f}%")
        
        plt.show()

if __name__ == "__main__":
    print("开始悬臂梁有限元分析...")
    start_time = time.time()
    
    # 创建求解器实例（可调整参数）
    solver = FEMCantileverBeamSolver(
        L=10.0,    # 梁长
        h=1.0,    # 梁高
        nx=50,    # x方向单元数
        ny=5,     # y方向单元数
        E=2.1e5,  # 弹性模量
        nu=0.3,   # 泊松比
        F=-1   # 末端力（负号表示向下）
    )
    
    # 生成网格
    solver.generate_mesh()
    print(f"网格生成完成: {solver.nel}个单元, {solver.nnp}个节点")
    
    # 组装系统
    solver.assemble_system()
    print(f"系统组装完成: 刚度矩阵尺寸 {solver.K.shape}")
    
    # 求解
    solver.solve()
    print("求解完成")
    
    # 绘制结果
    solver.plot_results()
    
    print(f"总运行时间: {time.time() - start_time:.2f}秒")
