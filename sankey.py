import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas_gbq

# Load the CSV file
#df = pd.read_csv('data_v2.csv')

query = """
SELECT *
FROM `dailymotion-data.users_kfeurprier.2025_supply_demand_sankey`
"""

# This will prompt you to authenticate in a browser the first time
df = pandas_gbq.read_gbq(query, project_id='dailymotion-data')

# For device_type == 'Web+App', set supply_source
df.loc[
    (df['supply_source'] == 'Audience Extension') & (df['device_type'] == 'Web+App'),
    'supply_source'
] = 'Audience Extension (Web+App)'

# For device_type == 'CTV', set supply_source
df.loc[
    (df['supply_source'] == 'Audience Extension') & (df['device_type'] == 'CTV'),
    'supply_source'
] = 'Audience Extension (CTV)'

#df['region'].fillna("N/A", inplace=True)
df['region'] = df['region'].fillna('N/A')

app = dash.Dash(__name__)

sorted_months = sorted(df['event_month'].unique())

month_options = [{'label': 'All', 'value': 'All'}] + \
                [{'label': m, 'value': m} for m in sorted_months]

app.layout = html.Div([
    html.Label("Region"),
    dcc.Dropdown(
        id='region-filter',
        options=[{'label': 'All', 'value': 'All'}] + 
                [{'label': r, 'value': r} for r in df['region'].unique()],
        value='All',
        clearable=True,
        placeholder="Select a region...",
    ),
    dcc.Dropdown(
        id='month-filter',
        options=month_options,
        value='All',
        clearable=True,
        placeholder="Select a month...",
    ),
    dcc.Graph(id='sankey-chart')
])

@app.callback(
    Output('sankey-chart', 'figure'),
    [Input('region-filter', 'value'),
     Input('month-filter', 'value')]
)

def update_sankey(region, month):
    import pandas as pd
    import plotly.graph_objects as go

    filtered_df = df.copy()
    title = "Sankey"

    # Apply filters
    if region and region != "All":
        filtered_df = filtered_df[filtered_df['region'] == region]
        title += f" | Region: {region}"
    else:
        title += " | All Regions"
    if month and month != "All":
        filtered_df = filtered_df[filtered_df['event_month'] == month]
        title += f" | Month: {month}"
    else:
        title += " | All Months"

    # Audience Extension types for breakdown
    ae_types = ["Audience Extension (Web+App)", "Audience Extension (CTV)"]

    # --- Calculate percentages (for demande_source and supply_source) ---
    total_revenue = filtered_df['revenue'].sum()
    demand_pct = (filtered_df.groupby('demande_source')['revenue'].sum() / total_revenue * 100).round(1)
    supply_pct = (filtered_df.groupby('supply_source')['revenue'].sum() / total_revenue * 100).round(1)

    # --- Node labels with line breaks and percentages ---
    def label_with_break(label, pct=None):
        return f"{label}<br>({pct:.1f}%)" if pct is not None else label

    demandes = filtered_df['demande_source'].unique().tolist()
    demandes_labels = [label_with_break(d, demand_pct[d]) if d in demand_pct else d for d in demandes]
    plateformes = filtered_df.loc[filtered_df['supply_source'].isin(ae_types), 'plateforme_AE'].unique().tolist()
    plateformes_labels = plateformes
    supplies = filtered_df['supply_source'].unique().tolist()
    supplies_labels = [label_with_break(s, supply_pct[s]) if s in supply_pct else s for s in supplies]

    all_labels = demandes_labels + plateformes_labels + supplies_labels
    label_indices = {label: i for i, label in enumerate(all_labels)}

    # Map original names to label versions
    demand_map = {d: l for d, l in zip(demandes, demandes_labels)}
    supply_map = {s: l for s, l in zip(supplies, supplies_labels)}

    # --- Links ---
    # 1. Audience Extension types: demande_source → plateforme_AE → supply_source
    ae_df = filtered_df[filtered_df['supply_source'].isin(ae_types)].copy()
    # demande_source → plateforme_AE
    links1_ae = ae_df.groupby(['demande_source', 'plateforme_AE'])['revenue'].sum().reset_index()
    links1_ae['source'] = links1_ae['demande_source'].map(demand_map).map(label_indices)
    links1_ae['target'] = links1_ae['plateforme_AE'].map(label_indices)
    # plateforme_AE → supply_source
    links2_ae = ae_df.groupby(['plateforme_AE', 'supply_source'])['revenue'].sum().reset_index()
    links2_ae['source'] = links2_ae['plateforme_AE'].map(label_indices)
    links2_ae['target'] = links2_ae['supply_source'].map(supply_map).map(label_indices)

    # 2. Non-Audience Extension: demande_source → supply_source
    non_ae_df = filtered_df[~filtered_df['supply_source'].isin(ae_types)].copy()
    links_non_ae = non_ae_df.groupby(['demande_source', 'supply_source'])['revenue'].sum().reset_index()
    links_non_ae['source'] = links_non_ae['demande_source'].map(demand_map).map(label_indices)
    links_non_ae['target'] = links_non_ae['supply_source'].map(supply_map).map(label_indices)

    # Combine all links
    sources_all = pd.concat([links1_ae['source'], links2_ae['source'], links_non_ae['source']])
    targets_all = pd.concat([links1_ae['target'], links2_ae['target'], links_non_ae['target']])
    values_all  = pd.concat([links1_ae['revenue'], links2_ae['revenue'], links_non_ae['revenue']])

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_labels,
        ),
        link=dict(
            source=sources_all,
            target=targets_all,
            value=values_all
        ))])

    fig.update_layout(
        title_text=title,
        font_size=14,         # You can also increase font size for readability
        width=1200,           # Increase width (default is 1000)
        height=700            # Increase height (default is 600)
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)