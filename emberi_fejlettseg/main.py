import os
import dash
from dash import html, dcc
from dash.dependencies import Output, Input
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

current_dir = os.path.dirname(os.path.abspath(__file__))
fejlettseg = pd.read_csv(os.path.join(current_dir, '1_emberi_fejlettseg.csv'))


def clean_data(df):
    #Numerikus oszlopok feltöltése
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Nem numerikus oszlopok feltöltése
    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns
    df[non_numeric_cols] = df[non_numeric_cols].fillna("Unknown")

    return df

fejlettseg = clean_data(fejlettseg)

fejlettseg_szintek = fejlettseg['Human Development Groups'].unique()

# Adatok előkészítése az 5. feladathoz
df_long = pd.melt(
    fejlettseg,
    id_vars=["Country"],
    var_name="Year",
    value_name="Human Development Index"
)

df_long["Year"] = df_long["Year"].str.extract(r"(\d{4})")
df_long = df_long.dropna(subset=["Year"])
df_long["Year"] = df_long["Year"].astype(int)

df_long = df_long.sort_values(by="Year")

#6.feladat előkészítés
years = [str(year) for year in range(1990, 2022)]

# Újrapróbálkozás átkonvertált adatszerkezettel
# df = pd.read_csv(os.path.join(current_dir, 'emberi_fejlettseg_jo.csv'))
#
# numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
# for col in numerical_columns:
#     df[col].fillna(df[col].mean(), inplace=True)
#
# categorical_columns = df.select_dtypes(include=['object']).columns
# for col in categorical_columns:
#     df[col].fillna('Unknown', inplace=True)
#
# df.drop_duplicates(inplace=True)


# A Layout
app.layout = html.Div(
    style={'backgroundColor': '#007D69', 'padding': '15px'},
    children=[

    html.H1('Emberi fejlettség elemzése',
        style={
            'color': 'white',
            'fontSize': '40px',
            'textAlign': 'center'
        }),

    # html.Div(
        # children=[
        #     html.Div(style={
        #         'width': '8px',
        #         'height': '8px',
        #         'backgroundColor': 'grey',
        #         'borderRadius': '50%',
        #         'display': 'inline-block',
        #         'verticalAlign': 'middle',
        #         'marginBottom': '80px'
        #     }),
        #     html.Div(style={
        #         'height': '1px',
        #         'backgroundColor': 'grey',
    #             'width': '80%',
    #             'display': 'inline-block',
    #             'verticalAlign': 'middle',
    #             'marginBottom': '80px'
    #         }),
    #         html.Div(style={
    #             'width': '8px',
    #             'height': '8px',
    #             'backgroundColor': 'grey',
    #             'borderRadius': '50%',
    #             'display': 'inline-block',
    #             'verticalAlign': 'middle',
    #             'marginBottom': '80px'
    #         }),
    #     ],
    #     style={'textAlign': 'center', 'marginTop': '20px'}
    # ),


        html.Div(
            children=[
                html.H3('Országok és helyezésük 2021-ben, fejlettségi szint alapján',
                    style={
                        'color': '#007D69'
                    }
                ),
                html.Div(id='selected-value'),

                dcc.Dropdown(
                    id='development-level-dropdown',
                    options=[{'label': szint, 'value': szint} for szint in fejlettseg_szintek],
                    value=None,
                    placeholder="Válassz fejlettségi szintet",
                ),

                html.Div(id='country-list', style={'marginTop': '20px'})
            ],
            style={
                'borderRadius': '15px',
                'margin': '40px',
                'padding': '40px',
                'backgroundColor': '#f0f0f0',
                'boxShadow': '5px 5px 10px rgba(0, 0, 0, 0.2)',
            }
        ),

        html.Div(
            children=[
                html.H3('Adott ország fejlettségi adatai',
                    style={
                        'color': '#007D69'
                    }
                ),

                dcc.Dropdown(
                    id='country-dropdown',
                    options=[{'label': country, 'value': country} for country in fejlettseg['Country'].unique()],
                    value=None,
                    placeholder="Válassz országot",
                ),
                html.Div(id='country-table'),
                html.P('Sajnos nem jött össze de kérlek nézd meg a kódot szerintem jó logikán indultam el.')
            ],
            style={
                'borderRadius': '15px',
                'margin': '40px',
                'padding': '40px',
                'backgroundColor': '#f0f0f0',
                'boxShadow': '5px 5px 10px rgba(0, 0, 0, 0.2)',
            }
        ),

        html.Div(
            children=[
                html.H3('Fejlettségi szint országok szerint évekre lebontva diagramon',
                        style={
                            'color': '#007D69'
                        }
                        ),

                dcc.Dropdown(
                    id='country-dropdown-5',
                    options=[{'label': country, 'value': country} for country in df_long['Country'].unique()],
                    multi=True,
                    placeholder="Válassz országokat"
                ),
                dcc.Graph(id='hdi-graph')


            ],
            style={
                'borderRadius': '15px',
                'margin': '40px',
                'padding': '40px',
                'backgroundColor': '#f0f0f0',
                'boxShadow': '5px 5px 10px rgba(0, 0, 0, 0.2)',
            }
        ),

        # html.Div(
        #     children=[
        #         html.H3('országok szerint mindjárt mindjárt valami',
        #                 style={
        #                     'color': '#007D69'
        #                 }
        #                 ),
        #
        #         dcc.Slider(
        #             id='year-slider',
        #             min=df['Years'].min(),
        #             max=df['Years'].max(),
        #             step=1,
        #             marks={year: str(year) for year in df['Years'].unique()},
        #             value=df['Years'].min(),
        #         ),
        #         # Legördülő lista a változó kiválasztásához
        #         dcc.Dropdown(
        #             id='variable-dropdown',
        #             options=[{'label': col, 'value': col} for col in df.columns if col not in ['Country', 'Years']],
        #             value='Human Development Index',  # alapértelmezett választás
        #             style={'width': '50%'}
        #         ),
        #         # Csúszka az osztályközök számának kiválasztásához
        #         dcc.Slider(
        #             id='bin-slider',
        #             min=5,
        #             max=50,
        #             step=1,
        #             marks={i: str(i) for i in range(5, 51, 5)},
        #             value=10,
        #         ),
        #         # A diagram, amely a kiválasztott adatokat mutatja
        #         dcc.Graph(id='histogram')
        #
        #
        #     ],
        #     style={
        #         'borderRadius': '15px',
        #         'margin': '40px',
        #         'padding': '40px',
        #         'backgroundColor': '#f0f0f0',
        #         'boxShadow': '5px 5px 10px rgba(0, 0, 0, 0.2)',
        #     }
        # ),
        html.Div(
            children=[
                html.H3('Gyakorisági diagram',
                        style={
                            'color': '#007D69'
                        }
                        ),

                # Év csúszka
                html.Div([
                    html.Label("Év választása:"),
                    dcc.Slider(
                        id='year-slider',
                        min=1990,
                        max=2021,
                        step=1,
                        value=2021,
                        marks={year: str(year) for year in range(1990, 2022)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'width': '80%', 'padding': '20px'}),

                # Változó legördülő lista
                html.Div([
                    html.Label("Változó választása:"),
                    dcc.Dropdown(
                        id='variable-dropdown',
                        options=[{'label': col, 'value': col} for col in fejlettseg.columns if
                                 'Life Expectancy' in col or 'Human Development Index' in col],
                        value='Human Development Index (2021)',  # Kezdő érték
                        style={'width': '50%'}
                    )
                ], style={'padding': '20px'}),

                # Osztályok száma csúszka
                html.Div([
                    html.Label("Osztályok száma:"),
                    dcc.Slider(
                        id='bins-slider',
                        min=5,
                        max=50,
                        step=1,
                        value=10,
                        marks={i: str(i) for i in range(5, 51, 5)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'width': '50%', 'padding': '20px'}),

                # Grafikák
                html.Div([
                    dcc.Graph(id='distribution-plot')
                ], style={'padding': '20px'})


            ],
            style={
                'borderRadius': '15px',
                'margin': '40px',
                'padding': '40px',
                'backgroundColor': '#f0f0f0',
                'boxShadow': '5px 5px 10px rgba(0, 0, 0, 0.2)',
            }
        ),

    html.Br(),
    html.Br(),
    dbc.Tabs([
        dbc.Tab([
            html.Ul([
                html.Br(),
                html.Li('Név: Petrikán Bianka'),
                html.Li('Neptun kód: JPW9G0'),

            ])
        ], label='Személyes adatok'),
        dbc.Tab([
            html.Ul([
                html.Br(),
                html.Li('A projekt célja: User interface-en keresztül elemezni és bemutatni az emberi fejlettséget különböző országokban és különböző mutatók szerint.'),
                html.Li('Időbeli lefedettség: 1990 - 2021'),
                html.Li('Frissítési gyakoriság: Évenként'),

            ])
        ], label='Projekt információk')
    ]),
])
#

#Proba dropdpwn
# @app.callback(
#     dash.dependencies.Output('selected-value', 'children'),
#     [dash.dependencies.Input('dropdown', 'value')]
# )
# def update_output(selected_value):
#     if selected_value:
#         return f'A választott érték: {selected_value}'
#     return "Nincs kiválasztva semmi."
#

#3. feladat dropdown callback
@app.callback(
    Output('country-list', 'children'),
    Input('development-level-dropdown', 'value')
)

def update_country_list(selected_szint):
    if selected_szint is None:
        return []
    else:
        filtered_data = fejlettseg[fejlettseg['Human Development Groups'] == selected_szint]
        countries_and_ranks = filtered_data[['Country', 'HDI Rank (2021)']].sort_values(by='Country')
        country_list = [f"{row['Country']} - {row['HDI Rank (2021)']}" for _, row in countries_and_ranks.iterrows()]
        return [html.Li(country) for country in country_list]

#4. feladat: táblázat
@app.callback(
    Output('country-table', 'children'),
    Input('country-dropdown', 'value')
)
def update_table(selected_country):
    if selected_country is None:
        return []
    else:
        country_data = fejlettseg[fejlettseg['Country'] == selected_country]

        date_columns = [col for col in country_data.columns if re.match(r"^.*\(\d{4}\)$", col)]

        indicator_columns = list(set([re.sub(r"\(\d{4}\)$", "", col) for col in date_columns]))

        years = [str(year) for year in range(1990, 2022)]

        data_for_table = []

        for indicator in indicator_columns:
            row = [indicator]
            for year in years:
                column_name = f"{indicator} ({year})"
                if column_name in date_columns:
                    row.append(country_data[column_name].values[0])
                else:
                    row.append(None)
            data_for_table.append(row)

        table = dbc.Table.from_dataframe(pd.DataFrame(data_for_table, columns=['Fejlettségi mutató'] + years), striped=True,
                                         bordered=True, hover=True)
        return table

#4. feladat próbáljuk újra


# 5. feladat

@app.callback(
    Output('hdi-graph', 'figure'),
    [Input('country-dropdown-5', 'value')]
)
def update_graph(selected_countries):
    if not selected_countries:
        return px.line(title="Nincs kiválasztott ország")

    # Szűrés a kiválasztott országokra
    filtered_df = df_long[df_long['Country'].isin(selected_countries)].copy()

    # Diagram készítése
    fig = px.line(
        filtered_df,
        x='Year',  # Évszámok az x tengelyen
        y='Human Development Index',  # HDI érték az y tengelyen
        color='Country',  # Országok színek szerint
        title="Human Development Index időszor országok szerint"
    )

    fig.update_layout(
        xaxis_title="Év",
        yaxis_title="Human Development Index (HDI)",
        legend_title="Országok"
    )

    return fig

#6. feladat
# @app.callback(
#     Output('histogram', 'figure'),
#     [Input('year-slider', 'value'),
#      Input('variable-dropdown', 'value'),
#      Input('bin-slider', 'value')]
# )
# def update_histogram(selected_year, selected_variable, selected_bins):
#
#     filtered_df = df[df['Years'] == selected_year]
#
#     data = filtered_df[selected_variable]
#
#     fig = px.histogram(filtered_df, x=selected_variable, nbins=selected_bins,
#                        title=f'{selected_variable} eloszlása {selected_year}-ban/ben')
#
#     # A diagram visszaadása
#     return fig

#6.feladat
@app.callback(
    Output('distribution-plot', 'figure'),
    [Input('year-slider', 'value'),
     Input('variable-dropdown', 'value'),
     Input('bins-slider', 'value')]
)
def update_plot(selected_year, selected_variable, bins):
    # Szűrés az adatok alapján
    filtered_df = df[['Country', selected_variable, str(selected_year)]].dropna()

    # Nevezze át a kiválasztott oszlopot a megjelenítéshez
    filtered_df = filtered_df.rename(columns={str(selected_year): 'Year_Value'})

    # Ábrázolás
    fig = px.histogram(
        filtered_df, x='Year_Value', color='Country', nbins=bins,
        labels={'Year_Value': selected_variable},
        title=f"{selected_variable} eloszlása {selected_year} évben"
    )
    return fig


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)

