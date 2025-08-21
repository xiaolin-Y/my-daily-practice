import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import time

class FEMPlateSolver:
    def __init__(self, L=1.0, W=0.5, nx=10, ny=5, E=210e9, nu=0.3, ngp=2):
        # 几何和材料参数
        self.L = L          # 板长度(m)
        self.W = W          # 板宽度(m)
        self.nx = nx        # x方向单元数
        self.ny = ny        # y方向单元数
        self.E = E          # 弹性模量(Pa)
        self.nu = nu        # 泊松比
        self.ngp = ngp      # 积分点数
        
        # 计算派生参数
        self.nnx = nx + 1   # x方向节点数
        self.nny = ny + 1   # y方向节点数
        self.nnp = self.nnx * self.nny  # 总节点数
        self.nel = nx * ny  # 总单元数
        self.ndof = 2 * self.nnp  # 总自由度

    def generate_mesh(self):
        """生成规则矩形网格"""
        # 节点坐标
        x_coords = np.linspace(0, self.L, self.nnx)# 包含终点
        y_coords = np.linspace(0, self.W, self.nny)
        X, Y = np.meshgrid(x_coords, y_coords)
        self.x = X.flatten()
        self.y = Y.flatten()# 将坐标展平为一维数组
        
        # 单元连接关系
        self.elements = np.zeros((self.nel, 4), dtype=int)
        for j in range(self.ny):# 遍历y方向单元
            for i in range(self.nx):
                elem = j * self.nx + i
                n1 = j * self.nnx + i# 单元左下角节点索引
                self.elements[elem] = [n1, n1+1, n1+self.nnx+1, n1+self.nnx]

    def assemble_stiffness_matrix(self):
        """组装刚度矩阵"""
        # 材料矩阵
        self.D = self.E/(1-self.nu**2) * np.array([
            [1, self.nu, 0],
            [self.nu, 1, 0],
            [0, 0, (1-self.nu)/2]
        ])
        
        # 初始化刚度矩阵
        self.K = lil_matrix((self.ndof, self.ndof))
        
        # 高斯积分点和权重
        gp, gw = np.polynomial.legendre.leggauss(self.ngp)
        
        # 遍历所有单元
        for e in range(self.nel):
            nodes = self.elements[e]
            coords = np.column_stack((self.x[nodes], self.y[nodes]))
            Ke = np.zeros((8, 8))
            
            # 高斯积分
            for i in range(self.ngp):
                for j in range(self.ngp):
                    xi, eta = gp[i], gp[j]
                    weight = gw[i] * gw[j]
                    
                    # 形函数导数
                    dN = self.shape_derivatives(xi, eta)
                    
                    # 雅可比矩阵
                    J = dN @ coords
                    detJ = np.linalg.det(J)
                    invJ = np.linalg.inv(J)
                    
                    # 全局导数
                    dN_global = invJ.T @ dN
                    
                    # B矩阵
                    B = np.zeros((3, 8))
                    for n in range(4):
                        B[0, 2*n] = dN_global[0, n]
                        B[1, 2*n+1] = dN_global[1, n]
                        B[2, 2*n] = dN_global[1, n]
                        B[2, 2*n+1] = dN_global[0, n]
                    
                    # 单元刚度矩阵贡献
                    Ke += B.T @ self.D @ B * detJ * weight
            
            # 组装到全局矩阵
            dofs = self.get_dof_indices(nodes)
            for i in range(8):
                for j in range(8):
                    self.K[dofs[i], dofs[j]] += Ke[i, j]
        
        # 转换为CSR格式
        self.K = csr_matrix(self.K)

    def solve(self):
        """求解系统方程"""
        # 边界条件: 固定左侧边界
        fixed_nodes = np.where(self.x == 0)[0]# 找到x=0的节点索引
        fixed_dofs = np.sort(np.concatenate((2*fixed_nodes, 2*fixed_nodes+1)))# 固定自由度x、y索引
        
        # 精确解作为边界条件
        self.u_exact, self.v_exact = self.exact_solution(self.x, self.y)
        
        # 应用边界条件
        bc_values = np.zeros(self.ndof)
        bc_values[0::2] = self.u_exact# 偶数索引是x方向位移
        bc_values[1::2] = self.v_exact
        
        # 标记自由度和固定自由度
        free_dofs = np.setdiff1d(np.arange(self.ndof), fixed_dofs)
        
        # 求解
        K_ff = self.K[free_dofs, :][:, free_dofs]
        F = -self.K[free_dofs, :][:, fixed_dofs] @ bc_values[fixed_dofs]
        
        u_free = spsolve(K_ff, F)
        
        # 组合解
        self.u = np.zeros(self.ndof)
        self.u[free_dofs] = u_free
        self.u[fixed_dofs] = bc_values[fixed_dofs]
        
        # 提取位移
        self.u_node = self.u[0::2]
        self.v_node = self.u[1::2]

    def plot_results(self):
        """可视化结果"""
        # 位移场
        X = self.x.reshape(self.nny, self.nnx)
        Y = self.y.reshape(self.nny, self.nnx)
        
        plt.figure(figsize=(12, 10))
        
        # 有限元解
        plt.subplot(2, 2, 1)
        plt.contourf(X, Y, self.u_node.reshape(self.nny, self.nnx), 20)
        plt.colorbar()
        plt.title("FE Solution - u")
        
        plt.subplot(2, 2, 2)
        plt.contourf(X, Y, self.v_node.reshape(self.nny, self.nnx), 20)
        plt.colorbar()
        plt.title("FE Solution - v")
        
        # 精确解
        plt.subplot(2, 2, 3)
        plt.contourf(X, Y, self.u_exact.reshape(self.nny, self.nnx), 20)
        plt.colorbar()
        plt.title("Exact Solution - u")
        
        plt.subplot(2, 2, 4)
        plt.contourf(X, Y, self.v_exact.reshape(self.nny, self.nnx), 20)
        plt.colorbar()
        plt.title("Exact Solution - v")
        
        plt.tight_layout()
        
        # 变形网格
        scale = 0.1 * min(self.L, self.W) / max(np.max(np.abs(self.u_node)), np.max(np.abs(self.v_node)))
        x_def = self.x + scale * self.u_node
        y_def = self.y + scale * self.v_node
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        self.plot_mesh(self.x, self.y, 'k-')
        plt.title("Original Mesh")
        
        plt.subplot(1, 2, 2)
        self.plot_mesh(x_def, y_def, 'r-')
        plt.title(f"Deformed Mesh (Scale: {scale:.2f})")
        
        plt.tight_layout()
        plt.show()

    # 辅助函数
    @staticmethod
    def shape_derivatives(xi, eta):
        """形函数导数"""
        return 0.25 * np.array([
            [-(1-eta), (1-eta), (1+eta), -(1+eta)],
            [-(1-xi), -(1+xi), (1+xi), (1-xi)]
        ])

    def get_dof_indices(self, nodes):
        """获取单元自由度索引"""
        dofs = np.zeros(8, dtype=int)
        for n in range(4):
            dofs[2*n] = 2 * nodes[n]
            dofs[2*n+1] = 2 * nodes[n] + 1
        return dofs

    @staticmethod
    def exact_solution(x, y):
        """精确解"""
        u = x**3 + 3*x*y**2
        v = 2*x*y + x*y**2 + y**3
        return u, v

    def plot_mesh(self, x, y, line_style):
        """绘制网格，并添加单元号和节点号"""
        X = x.reshape(self.nny, self.nnx)
        Y = y.reshape(self.nny, self.nnx)
        
        # 绘制网格线
        for j in range(self.nny):
            plt.plot(X[j, :], Y[j, :], line_style)
        for i in range(self.nnx):
            plt.plot(X[:, i], Y[:, i], line_style)
        
        # 标注节点号
        for n in range(self.nnp):
            plt.text(x[n], y[n], f'N{n}', color='blue', fontsize=8,
                    ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # 标注单元号
        for e in range(self.nel):
            nodes = self.elements[e]
            # 计算单元中心位置
            xc = np.mean(x[nodes])
            yc = np.mean(y[nodes])
            plt.text(xc, yc, f'E{e}', color='red', fontsize=10,
                    ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        plt.axis('equal')

if __name__ == "__main__":
    print("Starting FEM analysis...")
    start_time = time.time()
    
    solver = FEMPlateSolver(nx=10, ny=5)  # 可调整网格密度
    solver.generate_mesh()
    print("Mesh generated")
    
    solver.assemble_stiffness_matrix()
    print("Stiffness matrix assembled")
    
    solver.solve()
    print("Solution completed")
    
    solver.plot_results()
    
    print(f"Total execution time: {time.time()-start_time:.2f} seconds")