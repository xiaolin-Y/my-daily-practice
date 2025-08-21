import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import time
import math

class FEMPlateSolver:
    def __init__(self, L=1.0, W=1, nx=40, ny=40, E=210e3, nu=0.3, ngp=2):
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
        x_coords = np.linspace(0, self.L, self.nnx)
        y_coords = np.linspace(0, self.W, self.nny)
        X, Y = np.meshgrid(x_coords, y_coords)
        self.x = X.flatten()
        self.y = Y.flatten()
        
        # 单元连接关系（逆时针编号：左下、右下、右上、左上）
        self.elements = np.zeros((self.nel, 4), dtype=int)
        for j in range(self.ny):
            for i in range(self.nx):
                elem = j * self.nx + i
                n1 = j * self.nnx + i
                self.elements[elem] = [n1, # 左下
                                       n1+1, # 右下
                                       n1+self.nnx+1, # 右上
                                       n1+self.nnx]

    def assemble_system(self):
        """组装刚度矩阵和载荷向量"""
        # 材料矩阵 (平面应力)
        self.D = (self.E/(1-self.nu**2)) * np.array([
            [1, self.nu, 0],
            [self.nu, 1, 0],
            [0, 0, (1-self.nu)/2]
        ])
        
        # 初始化刚度矩阵和载荷向量
        self.K = lil_matrix((self.ndof, self.ndof))
        self.F = np.zeros(self.ndof)
        
        # 高斯积分点和权重
        gp, gw = np.polynomial.legendre.leggauss(self.ngp)
        
        # 遍历所有单元
        for e in range(self.nel):
            nodes = self.elements[e]
            coords = np.column_stack((self.x[nodes], self.y[nodes]))
            Ke = np.zeros((8, 8))
            Fe = np.zeros(8)
            
            # 高斯积分
            for i in range(self.ngp):
                for j in range(self.ngp):
                    xi, eta = gp[i], gp[j]
                    weight = gw[i] * gw[j]
                    
                    # 形函数及其导数
                    N = self.shape_functions(xi, eta)
                    dN = self.shape_derivatives(xi, eta)
                    
                    # 雅可比矩阵
                    J = dN @ coords
                    detJ = np.linalg.det(J)
                    if detJ <= 0:
                        raise ValueError(f"单元 {e} 雅可比行列式非正 ({detJ})，节点编号可能错误")
                    invJ = np.linalg.inv(J)
                    
                    # 全局导数
                    dN_global = invJ.T @ dN
                    
                    # B矩阵
                    B = np.zeros((3, 8))
                    for n in range(4):
                        B[0, 2*n] = dN_global[0, n]# ε_xx: ∂u/∂x
                        B[1, 2*n+1] = dN_global[1, n]# ε_yy: ∂v/∂y
                        B[2, 2*n] = dN_global[1, n]# γ_xy: ∂u/∂y
                        B[2, 2*n+1] = dN_global[0, n]# γ_xy: ∂v/∂x
                    
                    # 单元刚度矩阵贡献
                    Ke += B.T @ self.D @ B * detJ * weight
                    
                    # 单元载荷向量贡献 (基于精确解的体力)
                    x_phys = N @ coords[:, 0]
                    y_phys = N @ coords[:, 1]
                    bx, by = self.body_force(x_phys, y_phys)
                    
                    for n in range(4):
                        Fe[2*n] += N[n] * bx * detJ * weight
                        Fe[2*n+1] += N[n] * by * detJ * weight
            
            # 组装到全局矩阵和向量
            dofs = self.get_dof_indices(nodes)
            for i in range(8):
                self.F[dofs[i]] += Fe[i]
                for j in range(8):
                    self.K[dofs[i], dofs[j]] += Ke[i, j]
        
        # 转换为CSR格式（一种用于高效存储和操作稀疏矩阵的数据结构）
        self.K = csr_matrix(self.K)

    def solve(self):
        """求解系统方程"""
        # 计算精确解作为边界条件
        self.u_exact, self.v_exact = self.exact_solution(self.x, self.y)
        
        # 边界条件: 固定所有边界节点
        left_nodes = np.where(self.x == 0)[0]
        right_nodes = np.where(self.x == self.L)[0]
        bottom_nodes = np.where(self.y == 0)[0]
        top_nodes = np.where(self.y == self.W)[0]
        boundary_nodes = np.unique(np.concatenate([left_nodes, right_nodes, bottom_nodes, top_nodes]))
        fixed_dofs = np.sort(np.concatenate((2*boundary_nodes, 2*boundary_nodes+1)))
        
        # 应用边界条件(求解之前需要施加边界条件)
        bc_values = np.zeros(self.ndof)
        bc_values[0::2] = self.u_exact  # 偶数索引是x方向位移
        bc_values[1::2] = self.v_exact  # 奇数索引是y方向位移
        
        # 标记自由度和固定自由度
        free_dofs = np.setdiff1d(np.arange(self.ndof), fixed_dofs)
        
        # 求解
        K_ff = self.K[free_dofs, :][:, free_dofs]
        F_f = self.F[free_dofs] - self.K[free_dofs, :][:, fixed_dofs] @ bc_values[fixed_dofs]
        
        u_free = spsolve(K_ff, F_f)
        
        # 组合解
        self.u = np.zeros(self.ndof)
        self.u[free_dofs] = u_free
        self.u[fixed_dofs] = bc_values[fixed_dofs]
        
        # 提取位移
        self.u_node = self.u[0::2]
        self.v_node = self.u[1::2]

    def body_force(self, x, y):
        """根据新的精确解计算所需的体力"""
        pi = math.pi
        L = self.L
        W = self.W
        
        # 计算二阶导数
        d2u_dx2 = -(2*pi/L)**2 * np.sin(2*pi*x/L) * np.cos(2*pi*y/W)
        d2u_dy2 = -(2*pi/W)**2 * np.sin(2*pi*x/L) * np.cos(2*pi*y/W)
        d2u_dxdy = -(2*pi/L)*(2*pi/W) * np.cos(2*pi*x/L) * np.sin(2*pi*y/W)
        
        d2v_dx2 = -(2*pi/L)**2 * np.cos(2*pi*x/L) * np.sin(2*pi*y/W)
        d2v_dy2 = -(2*pi/W)**2 * np.cos(2*pi*x/L) * np.sin(2*pi*y/W)
        d2v_dxdy = -(2*pi/L)*(2*pi/W) * np.sin(2*pi*x/L) * np.cos(2*pi*y/W)
        
        # 计算体力分量
        bx = -self.E/(1-self.nu**2) * (d2u_dx2 + self.nu*d2v_dxdy + (1-self.nu)/2*(d2u_dy2 + d2v_dxdy))
        by = -self.E/(1-self.nu**2) * (self.nu*d2u_dxdy + d2v_dy2 + (1-self.nu)/2*(d2u_dxdy + d2v_dx2))
        
        return bx, by

    def shape_functions(self, xi, eta):
        """形函数"""
        return 0.25 * np.array([
            (1-xi)*(1-eta),
            (1+xi)*(1-eta),
            (1+xi)*(1+eta),
            (1-xi)*(1+eta)
        ])

    def shape_derivatives(self, xi, eta):
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

    def exact_solution(self, x, y):
        """精确解"""
        pi = math.pi
        L = self.L
        W = self.W
        u = np.sin(2*pi*x/L) * np.cos(2*pi*y/W)
        v = np.cos(2*pi*x/L) * np.sin(2*pi*y/W)
        return u, v

    def plot_results(self):
        """可视化结果"""
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
        plt.contourf(X, Y, self.u_exact.reshape(self.nny, self.nnx)-self.u_node.reshape(self.nny, self.nnx), 20)
        plt.colorbar()
        plt.title("relative error - u")
        
        plt.subplot(2, 2, 4)
        plt.contourf(X, Y, self.v_exact.reshape(self.nny, self.nnx), 20)
        plt.colorbar()
        plt.title("Exact Solution - v")
        
        plt.tight_layout()
        
        # 计算误差
        u_error = np.linalg.norm(self.u_node - self.u_exact)/np.linalg.norm(self.u_exact)
        v_error = np.linalg.norm(self.v_node - self.v_exact)/np.linalg.norm(self.v_exact)
        print(f"Relative error in u: {u_error*100:.2f}%")
        print(f"Relative error in v: {v_error*100:.2f}%")
        
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

    def plot_mesh(self, x, y, line_style):
        """绘制网格"""
        X = x.reshape(self.nny, self.nnx)
        Y = y.reshape(self.nny, self.nnx)
        
        for j in range(self.nny):
            plt.plot(X[j, :], Y[j, :], line_style)
        for i in range(self.nnx):
            plt.plot(X[:, i], Y[:, i], line_style)
        
        plt.axis('equal')

if __name__ == "__main__":
    print("Starting FEM analysis...")
    start_time = time.time()
    
    solver = FEMPlateSolver()
    solver.generate_mesh()
    print("Mesh generated")
    
    solver.assemble_system()
    print("System assembled")
    
    solver.solve()
    print("Solution completed")
    
    solver.plot_results()
    
    print(f"Total execution time: {time.time()-start_time:.2f} seconds")