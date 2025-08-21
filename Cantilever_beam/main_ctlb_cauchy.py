import numpy as np
import time
import FEM_ctlb_cauchy as fem1


if __name__ == "__main__":
    # 初始化参数
    params = fem1.BeamParameters()
    
    # 生成网格 (nx: x方向节点数, ny: y方向节点数)
    nel_x = 200 
    nel_y = 40 
    nodes, elements, nx, ny = fem1.generate_mesh(params, nx=nel_x + 1 , ny=nel_y+1)
    print(f"网格生成完成: {nodes.shape[0]}节点, {elements.shape[0]}四边形单元")
    
    # 有限元计算
    start_time = time.time()
    U_fem = fem1.fem_solution(nodes, elements, params)
    print(f"有限元计算完成: {time.time()-start_time:.2f}秒")
    
    # 计算误差
    u_fem = U_fem[::2]
    v_fem = U_fem[1::2]
    error = np.zeros_like(u_fem)
    for i, (x, y) in enumerate(nodes):
        u_ana, v_ana = fem1.analytical_solution(x, y, params)
        error[i] = np.sqrt((u_fem[i]-u_ana)**2 + (v_fem[i]-v_ana)** 2)
    print(f"最大位移误差: {np.max(error):.2e} m")
    
    # 可视化所有结果
    fem1.plot_results(nodes, elements, U_fem, params, nx, ny)
