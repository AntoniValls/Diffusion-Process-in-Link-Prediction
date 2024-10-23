from utils.evaluation import evaluate_all
import networkx as nx
from functools import partial
from diffusion.jackson_metrics import diffusion_centrality, Godfather
import argparse
from diffusion.complex_diffusion import paralell_complex_path


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--data_names', type=str, help="Comma-separated dataset names, e.g., 'Cora,PubMed'")
    args = parser.parse_args()
    

    eigen = partial(nx.eigenvector_centrality_numpy)
    degree = partial(nx.degree_centrality)
    diffusion = partial(diffusion_centrality, T=5)
    #complex_ = partial(paralell_complex_path, T=0.5) 
    
    model_name = args.model_name
    data_names = args.data_names.split(',') # Convert it into a list
    methods = [eigen, degree, diffusion]
    #, complex_]
    names = ["eigenvector_centrality",
         "degree_centrality", 
         "diffusion_centrality"]
         #"complex_path"]
    
    result = evaluate_all(model_name=model_name,
                          list_of_data=data_names,
                          method_list=methods,
                          method_names=names)
    
    # save outputs as csv
    for idx, df in enumerate(result):
        df.to_csv(f'data/evaluation/{model_name}/{data_names[idx]}.csv', index=False)
    
    
    



