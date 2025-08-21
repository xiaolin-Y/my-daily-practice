# main_energy_analysis.py
import add_energy_analysis as anal

if __name__ == "__main__":
    params = anal.BeamParameters()
    nodes, elements, nx, ny = anal.generate_mesh(params, 40, 20)  
    U_fem = anal.fem_solution(nodes, elements, params)
    anal.plot_results(nodes, elements, U_fem, params, nx, ny)
    