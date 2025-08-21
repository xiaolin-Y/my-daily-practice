import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def project_platehole():
    
    print("开始板孔有限元分析...")
    
    # 模型输入参数
    class Model:
        E = 1.02e5
        nu = 0.49999
        Tx = 10.0
        R = 0.3
        L = 1.0
        ndof = 2
        ncoord = 2
        eleType = 4
        nelem_1D = 20  # 网格划分密度，可调整以观察收敛性
    
    model = Model()
    
    # 网格生成
    (materialprops, ncoord, ndof, nnode, coords, nelem, maxnodes, connect,
     nelnodes, elident, nfix, fixnodes, ndload, dloads) = mesh_platehole(model)
    
    # 绘制初始网格检查
    plt.figure(figsize=(10, 8))
    plotmesh(coords, ncoord, nnode, connect, nelem, elident, nelnodes, 'b')
    
    # 添加节点编号
    for oi in range(nnode):
        plt.plot(coords[0, oi], coords[1, oi], 'r.', markersize=10)
        plt.text(coords[0, oi] + 0.05, coords[1, oi] + 0.05, 
                f'{oi+1}', fontsize=5)
    
    plt.title('Original Mesh and nNodes')
    plt.axis('equal')
    plt.grid(True)
    
    
    # K*dofs = r
    dofs = np.zeros(ndof * nnode)
    K = globalstiffness(ncoord, ndof, nnode, coords, nelem, maxnodes, 
                       elident, nelnodes, connect, materialprops, dofs)
    r = globaltraction(ncoord, ndof, nnode, ndload, coords, nelnodes, 
                      elident, connect, dloads, dofs)
    
    # 固定约束节点
    for n in range(nfix):
        rw = ndof * (fixnodes[0, n] - 1) + fixnodes[1, n] - 1
        for cl in range(ndof * nnode):
            K[rw, cl] = 0.0
            K[cl, rw] = 0.0  # 确保K矩阵对称
        K[rw, rw] = 1.0# 确保刚度矩阵对角线元素为1
        r[rw] = fixnodes[2, n]# 确保固定节点位移为0
    # 求解线性方程组（位移）
    dofs = np.linalg.solve(K, r)
    
    # 后处理 - 能量范数误差分析（应力）
    R = model.R
    Tx = model.Tx
    
    def f_srr(rr, th):
        return 0.5*Tx*(1.0-rr**2.0) + 0.5*Tx*(1.0-4.0*rr**2.0+3.0*rr**4.0)*np.cos(2.0*th)
    
    def f_stt(rr, th):
        return 0.5*Tx*(1.0+rr**2.0) - 0.5*Tx*(1.0+3.0*rr**4.0)*np.cos(2.0*th)
    
    def f_srt(rr, th):
        return -0.5*Tx*(1.0+2.0*rr**2.0-3.0*rr**4.0)*np.sin(2.0*th)
    
    enorm2 = 0.0  # 定义能量范数平方为0，后续会每个单元累加计算得到
    for lmn in range(nelem):# for循环可以遍历in后面的待处理数据集（序列），并单独地输出每个字符，存在临时变量lmn里
                            # 每个字符占一行，这个for循环内部的后续处理会遍历lmn，最后lmn会停在nelem-1的位置

        # 定义局部变量存储当前单元的节点坐标和位移数据
        lmncoord = np.zeros((ncoord, maxnodes))
        displacements = np.zeros((ndof, maxnodes))
        
        for a in range(nelnodes[lmn]):
            for i in range(ncoord):
                lmncoord[i, a] = coords[i, connect[a, lmn]]
            for i in range(ndof):
                displacements[i, a] = dofs[ndof * connect[a, lmn] + i]
        
        n = nelnodes[lmn]
        ident = elident[lmn]
        
        npoints = numberofintegrationpoints(ncoord, n)
        xilist = integrationpoints(ncoord, n, npoints)
        w = integrationweights(ncoord, n, npoints, ident)
        
        for intpt in range(npoints):
            xi = xilist[:, intpt]
            N = shapefunctions(n, ncoord, ident, xi)
            dNdxi = shapefunctionderivs(n, ncoord, ident, xi)
            
            # 计算积分点坐标
            x = np.zeros(ncoord)
            for i in range(ncoord):
                for a in range(n):
                    x[i] += lmncoord[i, a] * N[a]
            
            # 计算雅可比矩阵和行列式
            dxdxi = np.zeros((ncoord, ncoord))
            for i in range(ncoord):
                for j in range(ncoord):
                    for a in range(n):
                        dxdxi[i, j] += lmncoord[i, a] * dNdxi[a, j]
            
            dt = np.linalg.det(dxdxi)
            dxidx = np.linalg.inv(dxdxi)
            
            # 计算形函数对x的导数
            dNdx = np.zeros((n, ncoord))
            for a in range(n):
                for i in range(ncoord):
                    for j in range(ncoord):
                        dNdx[a, i] += dNdxi[a, j] * dxidx[j, i]
            
            # 计算应变
            strainf = np.zeros((ndof, ncoord))
            for i in range(ndof):
                for j in range(ncoord):
                    for a in range(n):
                        strainf[i, j] += 0.5 * (displacements[i, a] * dNdx[a, j] + 
                                               displacements[j, a] * dNdx[a, i])
            
            r_val = np.sqrt(x[0]**2.0 + x[1]**2.0)
            th = np.arctan(x[1] / x[0])

            srr = f_srr(R/r_val, th)
            stt = f_stt(R/r_val, th)
            srt = f_srt(R/r_val, th)
            
            RM = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])# 旋转矩阵
            stress_polar = np.array([[srr, srt], [srt, stt]])# 极坐标应力张量
            stresse = RM.T @ stress_polar @ RM# 转换为笛卡尔坐标系
            
            S = materialflexibility(ndof, ncoord, materialprops)# 材料柔度矩阵
            C = materialstiffness(ndof, ncoord, strainf, materialprops)# 材料刚度矩阵
            
            # 精确应变
            straine = np.zeros((ndof, ncoord))
            for i in range(ndof):
                for j in range(ncoord):
                    for k in range(ndof):
                        for l in range(ncoord):
                            straine[i, j] += S[i, j, k, l] * stresse[k, l]
            
            # 计算能量范数（∫Ω​(ϵ−ϵh​)TD(ϵ−ϵh​)dΩ）1/2
            for i in range(ndof):
                for j in range(ncoord):
                    for k in range(ndof):
                        for l in range(ncoord):
                            enorm2 += ((strainf[i, j] - straine[i, j]) * 
                                      C[i, j, k, l] * 
                                      (strainf[k, l] - straine[k, l]) * 
                                      w[intpt] * dt)
    
        print(f'能量范数平方: {enorm2:.6e}')
    
    # 后处理 - 应力分析
    sss = numberofintegrationpoints(ncoord, nelnodes[0])
    sigma = np.zeros((ncoord, ncoord, sss, nelem))
    
    for lmn in range(nelem):
        lmncoord = np.zeros((ncoord, maxnodes))# 存储当前单元（lmn）所有节点的坐标
        displacements = np.zeros((ndof, maxnodes))# 存储当前单元所有节点的位移数据
        
        for a in range(nelnodes[lmn]):
            for i in range(ncoord):
                lmncoord[i, a] = coords[i, connect[a, lmn]]# 提取节点坐标到 lmncoord
            for i in range(ndof):
                displacements[i, a] = dofs[ndof * connect[a, lmn] + i]# dofs 是全局位移数组；
                # displacements 是当前单元的局部位移数组
        
        n = nelnodes[lmn]
        ident = elident[lmn]
        npoints = numberofintegrationpoints(ncoord, n)# 二维四节点单元积分点数量
        
        # 设置积分点
        xilist = np.array([[-1.0, 1.0, 1.0, -1.0], [-1.0, -1.0, 1.0, 1.0]])
        
        for intpt in range(npoints):
            xi = xilist[:, intpt]
            N = shapefunctions(n, ncoord, ident, xi)
            dNdxi = shapefunctionderivs(n, ncoord, ident, xi)
            
            # 计算积分点坐标
            x = np.zeros(ncoord)
            for i in range(ncoord):
                for a in range(n):
                    x[i] += lmncoord[i, a] * N[a]
            
            # 计算雅可比矩阵
            dxdxi = np.zeros((ncoord, ncoord))
            for i in range(ncoord):
                for j in range(ncoord):
                    for a in range(n):
                        dxdxi[i, j] += lmncoord[i, a] * dNdxi[a, j]
            
            dxidx = np.linalg.inv(dxdxi)
            
            # 计算形函数对x的导数
            dNdx = np.zeros((n, ncoord))
            for a in range(n):
                for i in range(ncoord):
                    for j in range(ncoord):
                        dNdx[a, i] += dNdxi[a, j] * dxidx[j, i]
            
            # 计算应变
            strain = np.zeros((ndof, ncoord))
            for i in range(ndof):
                for j in range(ncoord):
                    for a in range(n):
                        strain[i, j] += 0.5 * (displacements[i, a] * dNdx[a, j] + 
                                              displacements[j, a] * dNdx[a, i])
            
            lmnstress = materialstress(ndof, ncoord, strain, materialprops)# 材料应力计算函数
            sigma[:, :, intpt, lmn] = lmnstress
    
    # 绘制应力等值线
    stresscontourplot(coords, nelem, nelnodes, connect, sigma)
    convergence_analysis()

def mesh_platehole(Model):
    """网格生成函数"""
    E = Model.E
    nu = 0.4999 
    Tx = Model.Tx
    R = Model.R
    L = Model.L
    ndof = Model.ndof
    ncoord = Model.ncoord
    maxnodes = Model.eleType
    nelem_1D = Model.nelem_1D
    
    nnode_1D = nelem_1D + 1
    
    materialprops = np.array([0.5*E/(1.0+nu), nu, 1])# 存储材料属性的数组（剪切模量、泊松比、占位符）
    
    #网格划分
    nnode = nnode_1D**2 * 2 - nnode_1D# 最终得到整个模型的总节点数（*2是考虑对称，多加了一条边要减去）
    nelem = nelem_1D**2 * 2# 总单元数（*2是考虑45度对称）
    nelnodes = 4 * np.ones(nelem, dtype=int)# 每个单元的节点数（四节点四边形单元）
    elident = np.zeros(nelem, dtype=int)# 单元类型标识（四节点四边形单元）
    
    # 固定节点（位移边界条件）
    nfix = nnode_1D * 2# 需要施加固定约束的总节点自由度数量
    # fixnodes数组记录需要施加固定约束的节点编号、约束方向及约束值
    fixnodes = np.zeros((3, nfix), dtype=int)
    fixnodes[0, :nnode_1D] = np.arange(1, nnode_1D + 1)# 节点编号：1到nnode_1D
    fixnodes[0, nnode_1D:nfix] = np.arange(nnode - nnode_1D + 1, nnode + 1)# 节点编号：最后nnode_1D个节点
    fixnodes[1, :nnode_1D] = 2# 约束方向：y方向（2代表y轴）——并不是赋值
    fixnodes[1, nnode_1D:nfix] = 1# 约束方向：x方向（1代表x轴）——并不是赋值
    fixnodes[2, :] = 0# 所有约束的位移值均为0——这里才是赋值
    
    # 节点坐标
    th = np.linspace(0.0, np.pi/4.0, nnode_1D)# 生成从 0 到 π/4 的由nnode_1D均匀分割角度数组
    x_1, y_1, x_2, y_2 = [], [], [], []# x_1, y_1：存储模型主区域；x_2, y_2：存储对称区域
    
    for irow in range(nnode_1D):
        # 生成主区域径向节点（x_row, y_row）；
        # 起点始终在孔边缘（R处），终点延伸至平板边界（L），确保网格贴合板孔的圆形边界和矩形外边界
        x_row = np.linspace(np.cos(th[irow]) * R, L, nnode_1D)
        y_row = np.linspace(np.sin(th[irow]) * R, L/nelem_1D * irow, nnode_1D)
        # 将生成的x_row和y_row追加到列表x_1和y_1中
        x_1.extend(x_row)
        y_1.extend(y_row)
        # 生成对称区域径向节点（x_2, y_2）
        if irow != nnode_1D - 1:# 排除最后一行（避免对称后重复计算边界节点）
            x_2 = list(y_row) + x_2
            y_2 = list(x_row) + y_2
            # 对称区域的x_2和y_2是将主区域的y_row和x_row反转后追加到列表中
    
    coords = np.array([x_1 + x_2, y_1 + y_2])# 列表拼接成完整的节点坐标数组
    
    # 牵引力施加位置
    ndload = nelem_1D * 2# 两组牵引力边界
    dloads = np.zeros((4, ndload))
    #创建一个 4 行、ndload列的全零数组，每行含义如下：
    # 第0行：单元编号（从1开始）
    # 第1行：施加牵引力的面编号（2号面）
    # 第2行：x方向牵引力分量
    # 第3行：y方向牵引力分量
    dloads[0, :] = np.arange(nelem_1D, nelem + 1, nelem_1D)# 单元编号（从1开始）
    dloads[1, :] = 2# 所有牵引力均施加在单元的 “2 号面”
    
    # 精确解函数作力边界条件
    def f_srr(rr, th):
        return 0.5*Tx*(1.0-rr**2.0) + 0.5*Tx*(1.0-4.0*rr**2.0+3.0*rr**4.0)*np.cos(2.0*th)
    
    def f_stt(rr, th):
        return 0.5*Tx*(1.0+rr**2.0) - 0.5*Tx*(1.0+3.0*rr**4.0)*np.cos(2.0*th)
    
    def f_srt(rr, th):
        return -0.5*Tx*(1.0+2.0*rr**2.0-3.0*rr**4.0)*np.sin(2.0*th)
    
    for idloads in range(ndload):
        nn1 = nnode_1D * (idloads + 1) - 1  # 得到边界单元的第一个节点索引
        nn2 = nnode_1D * (idloads + 2) - 1  # 得到边界单元的第二个节点索引
        x = 0.5 * (coords[:, nn1] + coords[:, nn2])# 计算边界单元两个节点的中点坐标
        r = np.sqrt(x[0]**2.0 + x[1]**2.0)
        th = np.arctan(x[1] / x[0])
        fn = np.array([1, 0]) if idloads < nelem_1D else np.array([0, 1])
        
        srr = f_srr(R/r, th)
        stt = f_stt(R/r, th)
        srt = f_srt(R/r, th)
        
        RM = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        s = RM.T @ np.array([[srr, srt], [srt, stt]]) @ RM# 转换为笛卡尔坐标系的应力张量
        t = s @ fn
        
        dloads[2, idloads] = t[0]
        dloads[3, idloads] = t[1]
    
    # 连接矩阵
    rowcount = 0
    connect = np.zeros((4, nelem), dtype=int)
    
    for elementcount in range(nelem):
        connect[0, elementcount] = elementcount + rowcount
        connect[1, elementcount] = elementcount + rowcount + 1
        connect[2, elementcount] = elementcount + rowcount + nnode_1D + 1
        connect[3, elementcount] = elementcount + rowcount + nnode_1D
        if (elementcount + 1) % nelem_1D == 0:
            rowcount += 1
    
    return (materialprops, ncoord, ndof, nnode, coords, nelem, maxnodes, 
            connect, nelnodes, elident, nfix, fixnodes, ndload, dloads)

def convergence_analysis():
    """收敛率分析函数"""
    # 不同网格尺寸下的能量范数平方数据
    enormsq = np.array([3.1953e-06, 9.1422e-07, 2.3778e-07, 6.0071e-08])  # R=0.3测试数据
    enorm = np.sqrt(enormsq)
    lgenorm = np.log(enorm)
    lgmesh = np.log(1.0/np.array([10, 20, 40, 80]))
    ntests = len(lgmesh)
    
    plt.figure(figsize=(10, 6))
    plt.plot(lgmesh, lgenorm, '.--', linewidth=1.5, markersize=20)
    plt.grid(True)
    plt.title('energy erro and mesh size (log-log)')
    plt.xlabel('mesh size')
    plt.ylabel('energy error')
    
    
    k = np.zeros(ntests - 1)
    for i in range(ntests - 1):
        k[i] = (lgenorm[i+1] - lgenorm[i]) / (lgmesh[i+1] - lgmesh[i])
    
    print('convergence_rates:', k)

def numberofintegrationpoints(ncoord, nelnodes, elident=None):
    """积分点数量"""
    if ncoord == 1:
        return nelnodes
    elif ncoord == 2:
        if nelnodes == 4:
            return 4

def integrationpoints(ncoord, nelnodes, npoints, elident=None):
    """积分点坐标"""
    xi = np.zeros((ncoord, npoints))
    
    if ncoord == 1:
        if npoints == 1:
            xi[0, 0] = 0.0
        elif npoints == 2:
            xi[0, 0] = -0.5773502692
            xi[0, 1] = -xi[0, 0]
        elif npoints == 3:
            xi[0, 0] = -0.7745966692
            xi[0, 1] = 0.0
            xi[0, 2] = -xi[0, 0]
    elif ncoord == 2:
        if nelnodes == 4:
            if npoints == 1:
                xi[0, 0] = 0.0
                xi[1, 0] = 0.0
            elif npoints == 4:
                xi[0, 0] = -0.5773502692
                xi[1, 0] = xi[0, 0]
                xi[0, 1] = -xi[0, 0]
                xi[1, 1] = xi[0, 0]
                xi[0, 2] = xi[0, 0]
                xi[1, 2] = -xi[0, 0]
                xi[0, 3] = -xi[0, 0]
                xi[1, 3] = -xi[0, 0]
    
    return xi

def integrationweights(ncoord, nelnodes, npoints, elident):
    """积分权重"""
    w = np.zeros(npoints)
    
    if ncoord == 1:
        if npoints == 1:
            w[0] = 2.0
        elif npoints == 2:
            w = np.array([1.0, 1.0])
        elif npoints == 3:
            w = np.array([0.555555555, 0.888888888, 0.555555555])
    elif ncoord == 2:
        if nelnodes == 4:
            if npoints == 1:
                w[0] = 4.0
            elif npoints == 4:
                w = np.array([1.0, 1.0, 1.0, 1.0])
            elif npoints == 9:
                w1D = np.array([0.555555555, 0.888888888, 0.555555555])
                for j in range(3):
                    for i in range(3):
                        n = 3 * j + i
                        w[n] = w1D[i] * w1D[j]
    
    return w

def shapefunctions(nelnodes, ncoord, elident, xi):
    """形函数"""
    N = np.zeros(nelnodes)
    
    if ncoord == 1:
        if nelnodes == 2:
            N[0] = 0.5 * (1.0 + xi[0])
            N[1] = 0.5 * (1.0 - xi[0])
    elif ncoord == 2:
        if nelnodes == 4:
            N[0] = 0.25 * (1.0 - xi[0]) * (1.0 - xi[1])
            N[1] = 0.25 * (1.0 + xi[0]) * (1.0 - xi[1])
            N[2] = 0.25 * (1.0 + xi[0]) * (1.0 + xi[1])
            N[3] = 0.25 * (1.0 - xi[0]) * (1.0 + xi[1])
    
    return N

def shapefunctionderivs(nelnodes, ncoord, elident, xi):
    """形函数导数"""
    dNdxi = np.zeros((nelnodes, ncoord))
    
    if ncoord == 1:
        if nelnodes == 2:
            dNdxi[0, 0] = 0.5
            dNdxi[1, 0] = -0.5
        elif nelnodes == 3:
            dNdxi[0, 0] = -0.5 + xi[0]
            dNdxi[1, 0] = 0.5 + xi[0]
            dNdxi[2, 0] = -2.0 * xi[0]
    elif ncoord == 2:
        if nelnodes == 4:
            dNdxi[0, 0] = -0.25 * (1.0 - xi[1])
            dNdxi[0, 1] = -0.25 * (1.0 - xi[0])
            dNdxi[1, 0] = 0.25 * (1.0 - xi[1])
            dNdxi[1, 1] = -0.25 * (1.0 + xi[0])
            dNdxi[2, 0] = 0.25 * (1.0 + xi[1])
            dNdxi[2, 1] = 0.25 * (1.0 + xi[0])
            dNdxi[3, 0] = -0.25 * (1.0 + xi[1])
            dNdxi[3, 1] = 0.25 * (1.0 - xi[0])
    
    return dNdxi

def nfacenodes(ncoord, nelnodes, elident, face):
    """单元面节点数"""
    if ncoord == 2:
        if nelnodes == 4:
            return 2

def facenodes(ncoord, nelnodes, elident, face):
    """单元面节点索引"""
    i4 = [1, 2, 3, 0]  # Python索引从0开始调整
    list_nodes = np.zeros(nfacenodes(ncoord, nelnodes, elident, face), dtype=int)
    
    if ncoord == 2:
        if nelnodes == 4:
            list_nodes[0] = face
            list_nodes[1] = i4[face]
    
    return list_nodes

def eldload(ncoord, ndof, nfacenodes, elident, coords, traction):
    """单元牵引力向量"""
    npoints = numberofintegrationpoints(ncoord - 1, nfacenodes)
    r = np.zeros(ndof * nfacenodes)
    xilist = integrationpoints(ncoord - 1, nfacenodes, npoints)
    w = integrationweights(ncoord - 1, nfacenodes, npoints, elident)
    
    for intpt in range(npoints):
        xi = xilist[:, intpt]
        N = shapefunctions(nfacenodes, ncoord - 1, elident, xi)
        dNdxi = shapefunctionderivs(nfacenodes, ncoord - 1, elident, xi)
        
        # 计算雅可比矩阵和行列式
        dxdxi = np.zeros((ncoord, ncoord - 1))
        for i in range(ncoord):
            for j in range(ncoord - 1):
                for a in range(nfacenodes):
                    dxdxi[i, j] += coords[i, a] * dNdxi[a, j]
        
        if ncoord == 2:
            dt = np.sqrt(dxdxi[0, 0]**2.0 + dxdxi[1, 0]**2.0)
        
        for a in range(nfacenodes):
            for i in range(ndof):
                row = ndof * a + i
                r[row] += N[a] * traction[i] * w[intpt] * dt
    
    return r

def globalstiffness(ncoord, ndof, nnode, coords, nelem, maxnodes, elident, 
                   nelnodes, connect, materialprops, dofs):
    """全局刚度矩阵组装"""
    Stif = np.zeros((ndof * nnode, ndof * nnode))
    
    for lmn in range(nelem):
        lmncoord = np.zeros((ncoord, maxnodes))
        lmndof = np.zeros((ndof, maxnodes))
        
        for a in range(nelnodes[lmn]):
            for i in range(ncoord):
                lmncoord[i, a] = coords[i, connect[a, lmn]]
            for i in range(ndof):
                lmndof[i, a] = dofs[ndof * connect[a, lmn] + i]
        
        n = nelnodes[lmn]
        ident = elident[lmn]
        kel = elstif(ncoord, ndof, n, ident, lmncoord, materialprops, lmndof)
        
        # 组装当前单元刚度矩阵到全局刚度矩阵
        for a in range(nelnodes[lmn]):
            for i in range(ndof):
                for b in range(nelnodes[lmn]):
                    for k in range(ndof):
                        rw = ndof * connect[a, lmn] + i
                        cl = ndof * connect[b, lmn] + k
                        Stif[rw, cl] += kel[ndof * a + i, ndof * b + k]
    
    return Stif

def globaltraction(ncoord, ndof, nnodes, ndload, coords, nelnodes, elident, 
                  connect, dloads, dofs):
    """全局牵引力向量组装"""
    r = np.zeros(ndof * nnodes)
    
    for load in range(ndload):
        # 提取相应单元面上节点坐标
        lmn = int(dloads[0, load]) - 1  # 转换为Python索引
        face = int(dloads[1, load]) - 1
        n = nelnodes[lmn]
        ident = elident[lmn]
        nfnodes = nfacenodes(ncoord, n, ident, face)
        nodelist = facenodes(ncoord, n, ident, face)
        
        lmncoord = np.zeros((ncoord, nfnodes))
        lmndof = np.zeros((ndof, nfnodes))
        
        for a in range(nfnodes):
            for i in range(ncoord):
                lmncoord[i, a] = coords[i, connect[nodelist[a], lmn]]
            for i in range(ndof):
                lmndof[i, a] = dofs[ndof * connect[nodelist[a], lmn] + i]
        
        # 计算单元载荷向量
        traction = np.zeros(ndof)
        for i in range(ndof):
            traction[i] = dloads[i + 2, load]
        
        rel = eldload(ncoord, ndof, nfnodes, ident, lmncoord, traction)
        
        # 组装当前单元牵引力向量到全局牵引力向量
        for a in range(nfnodes):
            for i in range(ndof):
                rw = ndof * connect[nodelist[a], lmn] + i
                r[rw] += rel[a * ndof + i]
    
    return r

def plotmesh(coords, ncoord, nnode, connect, nelem, elident, nelnodes, color):
    """绘制网格"""
    if ncoord == 2:
        for lmn in range(nelem):
            x = np.zeros((nelnodes[lmn], 2))
            for i in range(nelnodes[lmn]):
                x[i, :] = coords[:, connect[i, lmn]]
            
            if nelnodes[lmn] == 4:
                poly = Polygon(x, fill=False, edgecolor=color)
                plt.gca().add_patch(poly)
    
    plt.axis('equal')

def materialstress(ndof, ncoord, strain, materialprops):
    """计算应力"""
    C = materialstiffness(ndof, ncoord, strain, materialprops)
    stress = np.zeros((ndof, ncoord))
    
    for i in range(ndof):
        for j in range(ncoord):
            for k in range(ndof):
                for l in range(ncoord):
                    stress[i, j] += C[i, j, k, l] * strain[k, l]
    
    return stress

def materialstiffness(ndof, ncoord, strain, materialprops):
    """弹性张量"""
    mu = materialprops[0]
    nu = materialprops[1]
    planestrain = materialprops[2]
    
    C = np.zeros((ndof, ncoord, ndof, ncoord))
    
    for i in range(ndof):
        for j in range(ncoord):
            for k in range(ndof):
                for l in range(ncoord):
                    if planestrain == 1:
                        if i == j and k == l:
                            C[i, j, k, l] += 2.0 * mu * nu / (1.0 - 2.0 * nu)  # lambda
                    else:
                        if i == j and k == l:
                            C[i, j, k, l] += 2.0 * mu * nu / (1.0 - nu)  # lambda
                    
                    if i == l and k == j:
                        C[i, j, k, l] += mu
                    if i == k and j == l:
                        C[i, j, k, l] += mu
    
    return C

def elstif(ncoord, ndof, nelnodes, elident, coord, materialprops, displacement):
    """单元刚度矩阵组装"""
    npoints = numberofintegrationpoints(ncoord, nelnodes, elident)
    kel = np.zeros((ndof * nelnodes, ndof * nelnodes))
    
    # 设置积分点和权重
    xilist = integrationpoints(ncoord, nelnodes, npoints, elident)
    w = integrationweights(ncoord, nelnodes, npoints, elident)
    
    for intpt in range(npoints):
        xi = xilist[:, intpt]
        N = shapefunctions(nelnodes, ncoord, elident, xi)
        dNdxi = shapefunctionderivs(nelnodes, ncoord, elident, xi)
        
        # 计算雅可比矩阵和行列式
        dxdxi = np.zeros((ncoord, ncoord))
        for i in range(ncoord):
            for j in range(ncoord):
                for a in range(nelnodes):
                    dxdxi[i, j] += coord[i, a] * dNdxi[a, j]
        
        dxidx = np.linalg.inv(dxdxi)
        dt = np.linalg.det(dxdxi)
        
        # 计算形函数对x的导数
        dNdx = np.zeros((nelnodes, ncoord))
        for a in range(nelnodes):
            for i in range(ncoord):
                for j in range(ncoord):
                    dNdx[a, i] += dNdxi[a, j] * dxidx[j, i]
        
        # 计算应变张量
        strain = np.zeros((ndof, ncoord))
        for i in range(ndof):
            for j in range(ncoord):
                for a in range(nelnodes):
                    strain[i, j] += 0.5 * (displacement[i, a] * dNdx[a, j] + 
                                          displacement[j, a] * dNdx[a, i])
        
        # 计算材料切线刚度
        dsde = materialstiffness(ndof, ncoord, strain, materialprops)
        
        # 计算单元刚度矩阵
        for a in range(nelnodes):
            for i in range(ndof):
                for b in range(nelnodes):
                    for k in range(ndof):
                        row = ndof * a + i
                        col = ndof * b + k
                        for j in range(ncoord):
                            for l in range(ncoord):
                                kel[col, row] += (dsde[i, j, k, l] * 
                                                 dNdx[b, l] * dNdx[a, j] * 
                                                 w[intpt] * dt)
    
    return kel

def materialflexibility(ndof, ncoord, materialprops):
    """柔度张量"""
    mu = materialprops[0]
    nu = materialprops[1]
    lambda_val = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    temp1 = (lambda_val + 2.0 * mu) / (4.0 * lambda_val * mu + 4.0 * mu**2.0)
    temp2 = -lambda_val / (4.0 * lambda_val * mu + 4.0 * mu**2.0)
    temp3 = 0.5 / mu
    
    S = np.zeros((ndof, ncoord, ndof, ncoord))
    S[0, 0, 0, 0] = temp1
    S[1, 1, 1, 1] = temp1
    S[0, 0, 1, 1] = temp2
    S[1, 1, 0, 0] = temp2
    S[0, 1, 0, 1] = temp3
    S[1, 0, 1, 0] = temp3
    
    return S

def stresscontourplot(coords, nelem, nelnodes, connect, sigma):
    """应力等值线图"""
    temp = []
    
    # sigma11
    plt.figure(figsize=(10, 8))
    
    for lmn in range(nelem):
        s11 = np.sum(sigma[0, 0, :, lmn]) / 4.0
        temp.append(s11)
        
        x = np.zeros((nelnodes[lmn], 2))
        for i in range(nelnodes[lmn]):
            x[i, :] = coords[:, connect[i, lmn]]
        
        poly = Polygon(x, facecolor=plt.cm.viridis(s11/np.max(temp) if temp else 0), 
                      edgecolor='none')
        plt.gca().add_patch(poly)
    
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), 
                label='应力值')
    plt.title('stress on original mesh{11}')
    plt.axis('equal')
    plt.grid(False)
    
    print(f'最大应力值: {np.max(temp):.6f}')


if __name__ == "__main__":
    project_platehole()
    plt.show()