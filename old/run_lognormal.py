from utils.data_utils import fit_lognormal_distribution
import pandas as pd

# List of dataset combinations
datasets = [
    ('Planetoid', 'Cora'),
    ('Planetoid', 'CiteSeer'),
    ('Planetoid', 'PubMed'),
    ('AttributedGraphDataset', 'facebook'),
    ('AttributedGraphDataset', 'wiki'),
    ('LastFMAsia', 'LastFMAsia'),
    ('Twitch', 'EN'),
    ('Twitch', 'ES'),
    ('Twitch', 'DE'),
    ('CitationFull','Cora_ML'),
    ('CitationFull','DBLP'),
    ('Coauthor','CS'),
    ('Coauthor','Physics'),
    ('GitHub', 'GitHub')
]


results = []

# Loop over the datasets and calculate mu and sigma
for tgm_type, name in datasets:
    mu, sigma = fit_lognormal_distribution(tgm_type, name, plot=False)
    results.append({'tgm_type': tgm_type, 'name': name, 'mu': mu, 'sigma': sigma})

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Print the DataFrame
print(results_df)

# Write the DataFrame to a CSV file
results_df.to_csv('EDA/lognormal_fit_results.csv', index=False)

# %%
# Visualization
plt.figure(figsize=(14, 7))

# Scatter plot for mu values
plt.subplot(2, 1, 1)
sns.scatterplot(data=results_df, x='name', y='mu', s=100)
# sns.lineplot(data=results_df, x='name', y='sigma', color='green')
plt.title('Scatterplot of Mu Values')
plt.xlabel('Dataset')
plt.ylabel('Mu Value')
plt.xticks(rotation=45)

# Histograms for sigma values
plt.subplot(2, 1, 2)
sns.lineplot(data=results_df, x='name', y='sigma', color='green')
plt.title('Histograms of Sigma Values')
plt.xlabel('Dataset')
plt.ylabel('Sigma Value')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# %%
