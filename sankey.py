import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

# Load the CSV file
df = pd.read_csv('data_v1.csv')

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

df.fillna("N/A", inplace=True)

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
    title = "Sankey Chart"

    # Apply filters if needed
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

    # --- Calculate percentages ---
    total_revenue = filtered_df['revenue'].sum()
    supply_pct = (filtered_df.groupby('supply_source')['revenue'].sum() / total_revenue * 100).round(1)
    demand_pct = (filtered_df.groupby('demande_source')['revenue'].sum() / total_revenue * 100).round(1)

    # --- Node labels with percentages ---
    sources = filtered_df['supply_source'].unique().tolist()
    plateformes = filtered_df.loc[filtered_df['supply_source'].isin(ae_types), 'plateforme_AE'].unique().tolist()
    demandes = filtered_df['demande_source'].unique().tolist()
    
    # Helper for line breaks (retour à la ligne)
    def linebreak_label(label, pct=None):
        if pct is not None:
            return f"{label}<br>({pct:.1f}%)"
        return label

    sources_labels = [linebreak_label(s, supply_pct[s]) if s in supply_pct else s for s in sources]
    plateformes_labels = plateformes  # usually no percentage for intermediates
    demandes_labels = [linebreak_label(d, demand_pct[d]) if d in demand_pct else d for d in demandes]

    all_labels = sources_labels + plateformes_labels + demandes_labels
    label_indices = {label: i for i, label in enumerate(all_labels)}

    # Mapping original values to labels with percentages
    supply_map = {s: l for s, l in zip(sources, sources_labels)}
    demand_map = {d: l for d, l in zip(demandes, demandes_labels)}

    # --- Links ---
    # 1. Audience Extension types: supply_source → plateforme_AE → demande_source
    ae_df = filtered_df[filtered_df['supply_source'].isin(ae_types)].copy()
    # supply_source → plateforme_AE
    links1_ae = ae_df.groupby(['supply_source', 'plateforme_AE'])['revenue'].sum().reset_index()
    links1_ae['source'] = links1_ae['supply_source'].map(supply_map).map(label_indices)
    links1_ae['target'] = links1_ae['plateforme_AE'].map(label_indices)
    # plateforme_AE → demande_source
    links2_ae = ae_df.groupby(['plateforme_AE', 'demande_source'])['revenue'].sum().reset_index()
    links2_ae['source'] = links2_ae['plateforme_AE'].map(label_indices)
    links2_ae['target'] = links2_ae['demande_source'].map(demand_map).map(label_indices)

    # 2. Non-Audience Extension: supply_source → demande_source
    non_ae_df = filtered_df[~filtered_df['supply_source'].isin(ae_types)].copy()
    links_non_ae = non_ae_df.groupby(['supply_source', 'demande_source'])['revenue'].sum().reset_index()
    links_non_ae['source'] = links_non_ae['supply_source'].map(supply_map).map(label_indices)
    links_non_ae['target'] = links_non_ae['demande_source'].map(demand_map).map(label_indices)

    # Combine all links
    sources_all = pd.concat([links1_ae['source'], links2_ae['source'], links_non_ae['source']])
    targets_all = pd.concat([links1_ae['target'], links2_ae['target'], links_non_ae['target']])
    values_all  = pd.concat([links1_ae['revenue'], links2_ae['revenue'], links_non_ae['revenue']])

    # Sankey Plot
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

    fig.update_layout(title_text=title, font_size=10)
    return fig

if __name__ == '__main__':
    app.run(debug=True)