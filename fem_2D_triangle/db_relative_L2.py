import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import time
import math

class FEMPlateSolver:
    def __init__(self, L=1.0, W=1.0, nx=40, ny=40, E=210e3, nu=0.3, ngp=2):
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
        x_coords = np.linspace(0, self.L, self.nnx)
        y_coords = np.linspace(0, self.W, self.nny)
        X, Y = np.meshgrid(x_coords, y_coords)
        self.x = X.flatten()
        self.y = Y.flatten()
        
        self.elements = np.zeros((self.nel, 4), dtype=int)
        for j in range(self.ny):
            for i in range(self.nx):
                elem = j * self.nx + i
                n1 = j * self.nnx + i
                self.elements[elem] = [n1, n1+1, n1+self.nnx+1, n1+self.nnx]

    def assemble_system(self):
        """组装刚度矩阵和载荷向量"""
        self.D = (self.E/(1-self.nu**2)) * np.array([
            [1, self.nu, 0],
            [self.nu, 1, 0],
            [0, 0, (1-self.nu)/2]
        ])
        
        self.K = lil_matrix((self.ndof, self.ndof))
        self.F = np.zeros(self.ndof)
        gp, gw = np.polynomial.legendre.leggauss(self.ngp)
        
        for e in range(self.nel):
            nodes = self.elements[e]
            coords = np.column_stack((self.x[nodes], self.y[nodes]))
            Ke = np.zeros((8, 8))
            Fe = np.zeros(8)
            
            for i in range(self.ngp):
                for j in range(self.ngp):
                    xi, eta = gp[i], gp[j]
                    weight = gw[i] * gw[j]
                    
                    N = self.shape_functions(xi, eta)
                    dN = self.shape_derivatives(xi, eta)
                    
                    J = dN @ coords
                    detJ = np.linalg.det(J)
                    if detJ <= 0:
                        raise ValueError(f"单元 {e} 雅可比行列式非正 ({detJ})")
                    invJ = np.linalg.inv(J)
                    
                    dN_global = invJ.T @ dN
                    
                    B = np.zeros((3, 8))
                    for n in range(4):
                        B[0, 2*n] = dN_global[0, n]  # ε_xx: ∂u/∂x
                        B[1, 2*n+1] = dN_global[1, n]  # ε_yy: ∂v/∂y
                        B[2, 2*n] = dN_global[1, n]  # γ_xy: ∂u/∂y
                        B[2, 2*n+1] = dN_global[0, n]  # γ_xy: ∂v/∂x
                    
                    Ke += B.T @ self.D @ B * detJ * weight
                    
                    x_phys = N @ coords[:, 0]
                    y_phys = N @ coords[:, 1]
                    bx, by = self.body_force(x_phys, y_phys)
                    
                    for n in range(4):
                        Fe[2*n] += N[n] * bx * detJ * weight
                        Fe[2*n+1] += N[n] * by * detJ * weight
            
            dofs = self.get_dof_indices(nodes)
            for i in range(8):
                self.F[dofs[i]] += Fe[i]
                for j in range(8):
                    self.K[dofs[i], dofs[j]] += Ke[i, j]
        
        self.K = csr_matrix(self.K)

    def solve(self):
        """求解系统方程"""
        self.u_exact, self.v_exact = self.exact_solution(self.x, self.y)
        
        left_nodes = np.where(self.x == 0)[0]
        right_nodes = np.where(self.x == self.L)[0]
        bottom_nodes = np.where(self.y == 0)[0]
        top_nodes = np.where(self.y == self.W)[0]
        boundary_nodes = np.unique(np.concatenate([left_nodes, right_nodes, bottom_nodes, top_nodes]))
        fixed_dofs = np.sort(np.concatenate((2*boundary_nodes, 2*boundary_nodes+1)))
        
        bc_values = np.zeros(self.ndof)
        bc_values[0::2] = self.u_exact  
        bc_values[1::2] = self.v_exact  
        
        free_dofs = np.setdiff1d(np.arange(self.ndof), fixed_dofs)
        
        K_ff = self.K[free_dofs, :][:, free_dofs]
        F_f = self.F[free_dofs] - self.K[free_dofs, :][:, fixed_dofs] @ bc_values[fixed_dofs]
        
        u_free = spsolve(K_ff, F_f)
        
        self.u = np.zeros(self.ndof)
        self.u[free_dofs] = u_free
        self.u[fixed_dofs] = bc_values[fixed_dofs]
        
        self.u_node = self.u[0::2]
        self.v_node = self.u[1::2]

    def calculate_l2_error(self):
        """计算相对L2范数误差"""
        u_diff_squared = 0
        u_exact_squared = 0
        v_diff_squared = 0
        v_exact_squared = 0
        gp, gw = np.polynomial.legendre.leggauss(self.ngp)
        
        for e in range(self.nel):
            nodes = self.elements[e]
            coords = np.column_stack((self.x[nodes], self.y[nodes]))
            
            u_diff_e = self.u_node[nodes] - self.u_exact[nodes]
            v_diff_e = self.v_node[nodes] - self.v_exact[nodes]
            
            for i in range(self.ngp):
                for j in range(self.ngp):
                    xi, eta = gp[i], gp[j]
                    weight = gw[i] * gw[j]
                    
                    N = self.shape_functions(xi, eta)
                    dN = self.shape_derivatives(xi, eta)
                    J = dN @ coords
                    detJ = np.linalg.det(J)
                    
                    u_diff_point = N @ u_diff_e
                    v_diff_point = N @ v_diff_e
                    
                    x_phys = N @ coords[:, 0]
                    y_phys = N @ coords[:, 1]
                    u_exact_point, v_exact_point = self.exact_solution(x_phys, y_phys)
                    
                    u_diff_squared += (u_diff_point**2) * detJ * weight
                    u_exact_squared += (u_exact_point**2) * detJ * weight
                    v_diff_squared += (v_diff_point**2) * detJ * weight
                    v_exact_squared += (v_exact_point**2) * detJ * weight
        
        u_l2 = np.sqrt(u_diff_squared) / np.sqrt(u_exact_squared)
        v_l2 = np.sqrt(v_diff_squared) / np.sqrt(v_exact_squared)
        total_l2 = np.sqrt(u_diff_squared + v_diff_squared) / np.sqrt(u_exact_squared + v_exact_squared)
        
        return u_l2, v_l2, total_l2

    def calculate_energy_error(self):
        """计算能量误差: e_E = √(1/2 ∫(σ - σ_h) : (ε - ε_h) dΩ)"""
        integral = 0.0  # 存储∫(σ - σ_h) : (ε - ε_h) dΩ
        gp, gw = np.polynomial.legendre.leggauss(self.ngp)
        D = self.D  
        
        for e in range(self.nel):
            nodes = self.elements[e]
            coords = np.column_stack((self.x[nodes], self.y[nodes]))
            
            u_node_e = self.u_node[nodes]
            v_node_e = self.v_node[nodes]
            u_e = np.zeros(8)  
            for n in range(4):
                u_e[2*n] = u_node_e[n]
                u_e[2*n+1] = v_node_e[n]
            
            for i in range(self.ngp):
                for j in range(self.ngp):
                    xi, eta = gp[i], gp[j]
                    weight = gw[i] * gw[j]  
                    
                    N = self.shape_functions(xi, eta)
                    dN = self.shape_derivatives(xi, eta)  
                    
                    J = dN @ coords  
                    detJ = np.linalg.det(J)
                    if detJ <= 0:
                        raise ValueError(f"单元 {e} 雅可比行列式非正 ({detJ})")
                    invJ = np.linalg.inv(J)
                    
                    dN_global = invJ.T @ dN  
                    
                    B = np.zeros((3, 8))
                    for n in range(4):
                        B[0, 2*n] = dN_global[0, n]  # ε_xx = ∂u/∂x
                        B[1, 2*n+1] = dN_global[1, n]  # ε_yy = ∂v/∂y
                        B[2, 2*n] = dN_global[1, n]    # γ_xy = ∂u/∂y
                        B[2, 2*n+1] = dN_global[0, n]  # γ_xy = ∂v/∂x
                    
                    ε_h = B @ u_e  # 有限元应变: [ε_xx, ε_yy, γ_xy]
                    σ_h = D @ ε_h  # 有限元应力: [σ_xx, σ_yy, σ_xy]
                    
                    x_phys = N @ coords[:, 0]
                    y_phys = N @ coords[:, 1]
                    
                    # 精确位移偏导数（用于应变）
                    pi = math.pi
                    du_dx = (2*pi/self.L) * np.cos(2*pi*x_phys/self.L) * np.cos(2*pi*y_phys/self.W)
                    du_dy = -(2*pi/self.W) * np.sin(2*pi*x_phys/self.L) * np.sin(2*pi*y_phys/self.W)
                    dv_dx = -(2*pi/self.L) * np.sin(2*pi*x_phys/self.L) * np.sin(2*pi*y_phys/self.W)
                    dv_dy = (2*pi/self.W) * np.cos(2*pi*x_phys/self.L) * np.cos(2*pi*y_phys/self.W)
                    
                    # 精确应变（工程应变γ_xy）
                    ε_xx_exact = du_dx
                    ε_yy_exact = dv_dy
                    γ_xy_exact = du_dy + dv_dx
                    
                    # 精确应力（由材料矩阵和精确应变计算）
                    σ_xx_exact = D[0,0] * ε_xx_exact + D[0,1] * ε_yy_exact
                    σ_yy_exact = D[1,0] * ε_xx_exact + D[1,1] * ε_yy_exact
                    σ_xy_exact = D[2,2] * γ_xy_exact
                    
                    # 误差张量双点积项
                    dσ = np.array([
                        σ_xx_exact - σ_h[0],
                        σ_yy_exact - σ_h[1],
                        σ_xy_exact - σ_h[2]
                    ])
                    dε = np.array([
                        ε_xx_exact - ε_h[0],
                        ε_yy_exact - ε_h[1],
                        γ_xy_exact - ε_h[2]  # ε_h[2]是有限元γ_xy
                    ])
                    term = np.sum(dσ * dε)  # 双点积: (σ-σ_h):(ε-ε_h)
                    
                    # 累积积分（高斯积分近似）
                    integral += term * detJ * weight
        
        # 应用能量误差公式
        energy_error = np.sqrt(0.5 * integral)
        return energy_error

    def body_force(self, x, y):
        """根据精确解计算体力"""
        pi = math.pi
        L, W = self.L, self.W
        
        d2u_dx2 = -(2*pi/L)**2 * np.sin(2*pi*x/L) * np.cos(2*pi*y/W)
        d2u_dy2 = -(2*pi/W)**2 * np.sin(2*pi*x/L) * np.cos(2*pi*y/W)
        d2u_dxdy = -(2*pi/L)*(2*pi/W) * np.cos(2*pi*x/L) * np.sin(2*pi*y/W)
        
        d2v_dx2 = -(2*pi/L)**2 * np.cos(2*pi*x/L) * np.sin(2*pi*y/W)
        d2v_dy2 = -(2*pi/W)**2 * np.cos(2*pi*x/L) * np.sin(2*pi*y/W)
        d2v_dxdy = -(2*pi/L)*(2*pi/W) * np.sin(2*pi*x/L) * np.cos(2*pi*y/W)
        
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
        L, W = self.L, self.W
        u = np.sin(2*pi*x/L) * np.cos(2*pi*y/W)
        v = np.cos(2*pi*x/L) * np.sin(2*pi*y/W)
        return u, v

    def plot_results(self):
        """可视化结果"""
        X = self.x.reshape(self.nny, self.nnx)
        Y = self.y.reshape(self.nny, self.nnx)
        
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        plt.contourf(X, Y, self.u_node.reshape(self.nny, self.nnx), 20)
        plt.colorbar()
        plt.title("FE Solution - u")
        
        plt.subplot(2, 2, 2)
        plt.contourf(X, Y, self.v_node.reshape(self.nny, self.nnx), 20)
        plt.colorbar()
        plt.title("FE Solution - v")
        
        plt.subplot(2, 2, 3)
        plt.contourf(X, Y, (self.u_node - self.u_exact).reshape(self.nny, self.nnx), 20)
        plt.colorbar()
        plt.title("Error - u")
        
        plt.subplot(2, 2, 4)
        plt.contourf(X, Y, (self.v_node - self.v_exact).reshape(self.nny, self.nnx), 20)
        plt.colorbar()
        plt.title("Error - v")
        
        plt.tight_layout()
        
        u_l2, v_l2, total_l2 = self.calculate_l2_error()
        print(f"Relative L2 error in u: {u_l2*100:.2f}%")
        print(f"Relative L2 error in v: {v_l2*100:.2f}%")
        print(f"Relative Total L2 error: {total_l2*100:.2f}%")
        
        energy_error = self.calculate_energy_error()
        print(f"Energy error (e_E): {energy_error:.6e}")
        
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

    def plot_mesh(self, x, y, line_style):
        """绘制网格"""
        X = x.reshape(self.nny, self.nnx)
        Y = y.reshape(self.nny, self.nnx)
        
        for j in range(self.nny):
            plt.plot(X[j, :], Y[j, :], line_style)
        for i in range(self.nnx):
            plt.plot(X[:, i], Y[:, i], line_style)
        
        plt.axis('equal')

def plot_convergence_rate():
    """收敛率分析（含能量误差）"""
    grid_sizes = [5, 10, 20, 40, 80]  
    l2_errors = []
    energy_errors = []
    mesh_sizes = []  
    
    print("Running convergence study...")
    
    for nx in grid_sizes:
        print(f"Analyzing with nx=ny={nx}...")
        start_time = time.time()
        
        solver = FEMPlateSolver(nx=nx, ny=nx)
        solver.generate_mesh()
        solver.assemble_system()
        solver.solve()
        
        _, _, total_l2 = solver.calculate_l2_error()
        l2_errors.append(total_l2)
        
        energy_error = solver.calculate_energy_error()
        energy_errors.append(energy_error)
        
        h = max(solver.L / nx, solver.W / nx)  
        mesh_sizes.append(h)
        
        print(f"  L2 Error: {total_l2*100:.2f}%, Energy Error: {energy_error:.6e}, Mesh size: {h:.6f}")
        print(f"  Time: {time.time() - start_time:.2f} seconds")
    
    # 计算L2收敛率
    l2_rates = []
    for i in range(1, len(mesh_sizes)):
        error_ratio = l2_errors[i-1] / l2_errors[i]
        mesh_ratio = mesh_sizes[i-1] / mesh_sizes[i]
        if error_ratio > 0 and mesh_ratio > 0:
            l2_rate = np.log(error_ratio) / np.log(mesh_ratio)
            l2_rates.append(l2_rate)
    
    avg_l2_rate = np.mean(l2_rates) if l2_rates else 0
    
    # 计算能量误差收敛率
    energy_rates = []
    for i in range(1, len(mesh_sizes)):
        error_ratio = energy_errors[i-1] / energy_errors[i]
        mesh_ratio = mesh_sizes[i-1] / mesh_sizes[i]
        if error_ratio > 0 and mesh_ratio > 0:
            energy_rate = np.log(error_ratio) / np.log(mesh_ratio)
            energy_rates.append(energy_rate)
    
    avg_energy_rate = np.mean(energy_rates) if energy_rates else 0
    
    print(f"\nAverage L2 error convergence rate: {avg_l2_rate:.2f}")
    print(f"Average energy error convergence rate: {avg_energy_rate:.2f}")
    
    plt.figure(figsize=(10, 6))
    
    plt.loglog(mesh_sizes, l2_errors, 'o-', label=f'L2 Error (rate: {avg_l2_rate:.2f})')
    plt.loglog(mesh_sizes, energy_errors, 's-', label=f'Energy Error (rate: {avg_energy_rate:.2f})')
    
    ref_x = [mesh_sizes[0], mesh_sizes[-1]]
    ref_y1 = [l2_errors[0], l2_errors[0]*(ref_x[1]/ref_x[0])**1]
    ref_y2 = [l2_errors[0], l2_errors[0]*(ref_x[1]/ref_x[0])**2]
    
    plt.loglog(ref_x, ref_y1, '--', label='1st order')
    plt.loglog(ref_x, ref_y2, '-.', label='2nd order')
    
    plt.xlabel('Mesh size (h)')
    plt.ylabel('Error')
    plt.title('Convergence Analysis')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.tight_layout()

if __name__ == "__main__":
    print("Starting FEM analysis...")
    start_time = time.time()
    
    solver = FEMPlateSolver(nx=40, ny=40)
    solver.generate_mesh()
    print("Mesh generated")
    
    solver.assemble_system()
    print("System assembled")
    
    solver.solve()
    print("Solution completed")
    
    solver.plot_results()
    
    plot_convergence_rate()
    
    print(f"Total execution time: {time.time()-start_time:.2f} seconds")
    plt.show()
