from feon.sa import *
if __name__ == "__main__":
   
   k1 = 100
   k2 = 200

   n0 = Node(0,0)
   n1 = Node(1,0)
   n2 = Node(2,0)

   e0 = Spring1D11((n0,n1),k1)
   e1 = Spring1D11((n1,n2),k2)

   s = System()

   s.add_nodes(n0,n1,n2)
   s.add_elements(e0,e1)

   s.add_node_force(2,Fx = 15)
   s.add_fixed_sup(0)

   s.solve()
  
