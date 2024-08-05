from diffusion.all_centralities import run_all_centralities

p = 0.5             # Todo: run it for different variables   
T = 4
threshold = 2

# List of dataset combinations
datasets = [
    ('Planetoid', 'Cora'),
    ('Planetoid', 'CiteSeer'),
    ('Planetoid', 'PubMed'),
    ('AttributedGraphDataset', 'facebook'),
    ('AttributedGraphDataset', 'wiki'),
    ('LastFMAsia', 'LastFMAsia'),
    ('Twitch', 'EN'),
    ('CitationFull','cora_ml'),
    ('Twitch', 'ES'),
    ('Twitch', 'DE'),
    ('GitHub', 'GitHub'),
    ('CitationFull','DBLP'),
    ('Coauthor','CS'),
    ('Coauthor','Physics'),
    ('GitHub', 'GitHub')
]

# Dictionary to store results
results = {}

# Loop over the datasets and calculate mu and sigma
for tgm_type, name in datasets:
    run_all_centralities(tgm_type, name, p, T, threshold )


# %%
