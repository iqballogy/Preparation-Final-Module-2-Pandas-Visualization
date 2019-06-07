import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from plotly import tools
import numpy as np
from sqlalchemy import create_engine
import pandas as pd
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)
server = app.server


 #CREATE DATAFRAME FROM SQL
engine = create_engine(
    "mysql+mysqlconnector://root:abc123@localhost/latihan?host=localhost?port=3306")
conn = engine.connect()
results = conn.execute("SELECT * from mobil").fetchall()
dfMobil = pd.DataFrame(results, columns=results[0].keys())

def generate_table(dataframe, max_rows=10) :
    return html.Table(
         # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(str(dataframe.iloc[i,col])) for col in range(len(dataframe.columns))
        ]) for i in range(min(len(dataframe), max_rows))]
    )

app.title = 'Dashboard mpg'

app.layout = html.Div([
    html.H1('Cars Data Analytics'),
    html.H3('1870-1983'
    ),
        dcc.Tabs(id="tabs", value='tab-1', children=[
            dcc.Tab(label = 'Cars 1970 - 1982 Data', value = 'tab-1', children = [
                html.Div([
                    html.Div([
                        html.P('Find Car By: '),
                        dcc.Input(
                            id = 'filternametable',
                            type = 'text',
                            value = '',
                            style = dict(width='100%')
                        )
                    ], className = 'col-4'),
                    html.Div([
                        html.P('Filter Model Year : '),
                        dcc.Dropdown(
                             id='filterModelYear',
                             options=[i for i in [{ 'label': 'All year', 'value': '' },
                                                { 'label': '1970', 'value': '70' },
                                                { 'label': '1971', 'value': '71' },
                                                { 'label': '1972', 'value': '72' },
                                                { 'label': '1973', 'value': '73' },
                                                { 'label': '1974', 'value': '74' },
                                                { 'label': '1975', 'value': '75' },
                                                { 'label': '1976', 'value': '76' },
                                                { 'label': '1977', 'value': '77' },
                                                { 'label': '1978', 'value': '78' },
                                                { 'label': '1979', 'value': '79' },
                                                { 'label': '1980', 'value': '80' },
                                                { 'label': '1981', 'value': '81' },
                                                { 'label': '1982', 'value': '82' },]],
                            value=''
                        )
                    ], className = 'col-4'),
                    html.Div([
                        html.P('Category : '),
                        dcc.Dropdown(
                            id='filterOrigin',
                            options=[i for i in [{ 'label': 'All origin', 'value': '' },
                                                { 'label': 'USA', 'value': 'usa' },
                                                { 'label': 'Europe', 'value': 'europe' },                                                
                                                { 'label': 'Japan', 'value': 'japan' }]],
                            value=''
                        )
                    ], className='col-4') 
                ], className='row'),
                html.Br(),
                html.Div([
                    html.Div([
                        html.P('Select Range of MPG : '),
                        dcc.RangeSlider(
                            marks={i: str(i) for i in range(dfMobil['mpg'].min(), dfMobil['mpg'].max())},
                            min=dfMobil['mpg'].min(),
                            max=dfMobil['mpg'].max(),
                            value=[dfMobil['mpg'].min(),dfMobil['mpg'].max()],
                            className='rangeslider',
                            id='filtermpgslider'
                        )
                    ], className='col-9'),
                    html.Div([

                    ],className='col-1'),
                    html.Div([
                        html.Br(),
                        html.Button('Find Cars', id='buttonsearch', style=dict(width='100%'))
                    ], className='col-2')
                ], className='row'),
                html.Br(),html.Br(),html.Br(),
                html.Div([
                    html.Div([
                        html.P('Number of Display : '),
                        dcc.Input(
                            id='filterrowstable',
                            type='number',
                            value=10,
                            style=dict(width='100%')
                        )
                    ], className='col-1')
                ], className='row'),
                html.Center([
                    html.H2('Cars Data', className='title'),
                    html.Div(id='tablediv'),
                ])
            ]),
#TAB 2
        dcc.Tab(label='New Input', value='tab-2', children=[
            html.Div([
                html.Div([
                    html.P('Name : '),
                    dcc.Input(
                        id='inputnametable',
                        type='text',
                        value='',
                        style=dict(width='100%')
                    )
                ], className='col-4'),
                html.Div([
                    html.P('Input Model Year : '),
                    dcc.Dropdown(
                            id='inputModelYear',
                            options=[i for i in [{ 'label': 'All year', 'value': '' },
                                                { 'label': '1970', 'value': '70' },
                                                { 'label': '1971', 'value': '71' },
                                                { 'label': '1972', 'value': '72' },
                                                { 'label': '1973', 'value': '73' },
                                                { 'label': '1974', 'value': '74' },
                                                { 'label': '1975', 'value': '75' },
                                                { 'label': '1976', 'value': '76' },
                                                { 'label': '1977', 'value': '77' },
                                                { 'label': '1978', 'value': '78' },
                                                { 'label': '1979', 'value': '79' },
                                                { 'label': '1980', 'value': '80' },
                                                { 'label': '1981', 'value': '81' },
                                                { 'label': '1982', 'value': '82' },]],
                            value=''
                    )
                ], className = 'col-4'),
                html.Div([
                    html.P('Category : '),
                    dcc.Dropdown(
                        id='inputOrigin',
                        options=[i for i in [{ 'label': 'All origin', 'value': '' },
                                            { 'label': 'USA', 'value': 'usa' },
                                            { 'label': 'Europe', 'value': 'europe' },
                                            { 'label': 'Japan', 'value': 'japan' }]],
                        value=''
                    )
                ], className='col-4')
            ], className='row'),
            html.Br(),
            html.Div([
                html.Div([
                    html.P('Input Brand : '),
                    dcc.Dropdown(
                        id='inputBrand',
                        options=[{'label': i, 'value': i} for i in dfMobil['brand']],
                        value='nissan'
                    )
                ], className='col-3'),
                html.Div([
                ],className='col-1'),
                html.Div([
                    html.Br(),
                    html.Button('Input to Database', id='buttonInput', style=dict(width='100%'))
                ], className='col-2')
            ], className='row'),
            html.Br(),html.Br(),
            html.Center([
                html.H2('Recent Input', className='title'),
                html.Div(id='tableinput')
            ])
        ]),

#TAB 3
        dcc.Tab(label='Scatter Plot', value='tab-3', children=[
            html.Br(), html.Br(),
            html.Div([
                html.Div([
                    html.P('Hue : '),
                    dcc.Dropdown(
                        id='hueScatterPlot',
                        options=[i for i in [{ 'label': 'Origin', 'value': 'origin'},
                                            { 'label': 'Manufacturer', 'value': 'brand'},
                                            { 'label': 'Model Year', 'value': 'model_year'}]],
                        value='origin'
                    )
                ], className='col-4'),
                html.Div([
                    html.P('X : '),
                    dcc.Dropdown(
                        id='xplotscatter',
                        options=[{'label': i, 'value': i} for i in dfMobil.columns[6:]],
                        value='displacement'
                    )
                ], className='col-4'),
                html.Div([
                    html.P('Y : '),
                    dcc.Dropdown(
                        id='yplotscatter',
                        options=[{'label': i, 'value': i} for i in dfMobil.columns[6:]],
                        value='mpg'
                    )
                ], className='col-4')
            ], className='row'),
            html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
            dcc.Graph(
                id='scattergraph'
            )
        ]),

#TAB 4
        dcc.Tab(label='Pie Chart', value='tab-4', children=[
             html.Div([
                html.Div([
                    html.P('Group : '),
                    dcc.Dropdown(
                        id='groupplotpie',
                        options=[i for i in [{ 'label': 'Origin', 'value': 'origin'},
                                            { 'label': 'Manufacturer', 'value': 'brand'},
                                            { 'label': 'Model Year', 'value': 'model_year'}]],
                        value='origin'
                    )
                ], className='col-4')
            ], className='row'),
            html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
            dcc.Graph(
                id='piegraph'
            )
        ]),
#TAB 5
        dcc.Tab(label='Categorical Plots', value='tab-5', children=[
            html.Div([
                html.Div([
                    html.P('Kind : '),
                    dcc.Dropdown(
                        id='jenisplotcategory',
                        options=[{'label': i, 'value': i} for i in ['Bar','Box','Violin']],
                        value='Bar'
                    )
                ], className='col-3'),
                html.Div([
                    html.P('X : '),
                    dcc.Dropdown(
                        id='xplotcategory',
                        options=[{'label': 'Model Year', 'value': 'model_year'}],
                        value='model_year'
                    )
                ], className='col-3'),
                html.Div([
                    html.P('Y : '),
                    dcc.Dropdown(
                        id='yplotcategory',
                        options=[{'label': i, 'value': i} for i in dfMobil.columns[6:]],
                        value='mpg'
                    )
                ], className='col-3'),
                html.Div([
                    html.P('Stats : '),
                    dcc.Dropdown(
                        id='statsplotcategory',
                        options=[i for i in [{ 'label': 'Mean', 'value': 'mean' },
                                            { 'label': 'Standard Deviation', 'value': 'std' },
                                            { 'label': 'Count', 'value': 'count' },
                                            { 'label': 'Min', 'value': 'min' },
                                            { 'label': 'Max', 'value': 'max' },
                                            { 'label': '25th Percentiles', 'value': '25%' },
                                            { 'label': 'Median', 'value': '50%' },
                                            { 'label': '75th Percentiles', 'value': '75%' }]],
                        value='mean',
                        disabled=False
                    )
                ], className='col-3')
            ], className='row'),
            html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
            dcc.Graph(
                id='categorygraph'
            )
        ]),
        dcc.Tab(label='Histogram', value='tab-6', children=[
            html.Div([
                html.Div([
                    html.P('X : '),
                    dcc.Dropdown(
                        id='xplothist',
                        options=[{'label': i, 'value': i} for i in dfMobil.columns[6:]],
                        value='mpg'
                    )
                ], className='col-4'),
                html.Div([
                    html.P('Hue : '),
                    dcc.Dropdown(
                        id='hueplothist',
                        options=[i for i in [{ 'label': 'Origin', 'value': 'origin'},
                                            { 'label': 'Manufacturer', 'value': 'brand'},
                                            { 'label': 'Model Year', 'value': 'model_year'}]],
                        value='All'
                    )
                ], className='col-4'),
                html.Div([
                    html.P('Standard deviation : '),
                    dcc.Dropdown(
                        id='stdplothist',
                        options=[{'label': '{} Standard Deviation'.format (i), 'value': i} for i in ['1','2 ','3']],
                        value='2'
                    )
                ], className='col-4')
            ], className='row'),
            html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
            dcc.Graph(
                id='histgraph'
            )
        ])
    ]),
], style={
    'maxWidth': '1200px',
    'margin': '0 auto'
})

#CALLBACK TABLE

#CALLBACK TAB 1
@app.callback(
    Output(component_id='tablediv', component_property='children'),
    [Input('buttonsearch', 'n_clicks'),
    Input('filterrowstable', 'value')],
    [State('filternametable', 'value'),
    State('filterModelYear', 'value'),
    State('filterOrigin', 'value'),
    State('filtermpgslider', 'value')]
)
def update_table(n_clicks,maxrows, name,year,origin,mpg):
    dfFilter = dfMobil[(dfMobil['name'].str.contains(name)) & ((dfMobil['mpg'] >= mpg[0]) & (dfMobil['mpg'] <= mpg[1]))]
    if(year != '') :
        dfFilter = dfFilter[dfFilter['model_year'] == int(year)]
    if(origin != '') :
        dfFilter = dfFilter[dfFilter['origin'] == origin]

    return generate_table(dfFilter, max_rows=maxrows)

rowcolhist = {
    'All': { 'row': 1, 'col': 1 },
    'Generation': { 'row': 3, 'col': 2 },
    'Legendary': { 'row': 1, 'col': 2 }
}

#CALLBACK TAB 2
@app.callback(
    Output(component_id = 'tableinput', component_property='children'),
    [Input('buttonInput', 'n_clicks')],
    [State('inputnametable', 'value'),
    State('inputModelYear', 'value'),
    State('inputOrigin', 'value'),
    State('inputBrand', 'value')]
)

def inputdatabase(n_clicks, name, year,origin,brand):
    if name != '':
        conn.execute("INSERT INTO mobil VALUES(NULL,'{}','{}',{},'{}',{},{},{},{},{})".format(brand,name,year,origin,4,26,200,125,1522))
    results = conn.execute("SELECT * FROM mobil ORDER BY ID DESC LIMIT 5").fetchall()
    dfsql = pd.DataFrame(results, columns=results[0].keys())
    return generate_table(dfsql)


Scatterlegend = {
    'model_year': {i:i for i in dfMobil['model_year'].unique()},
    'brand': { i:i for i in dfMobil['brand'].unique()},
    'origin': { i:i for i in dfMobil['origin'].unique()}
}

#CALLBACK TAB 3
@app.callback(
    Output(component_id='scattergraph', component_property='figure'),
    [Input(component_id='hueScatterPlot', component_property='value'),
    Input(component_id='xplotscatter', component_property='value'),
    Input(component_id='yplotscatter', component_property='value')]
)
def callbackUpdateScatterGraph(hue,x,y) :
    return dict(
                data=[
                    go.Scatter(
                        x=dfMobil[dfMobil[hue] == val][x],
                        y=dfMobil[dfMobil[hue] == val][y],
                        name=Scatterlegend[hue][val],
                        mode='markers'
                    ) for val in dfMobil[hue].unique()
                ],
                layout=go.Layout(
                    title= 'Scatter Plot Cars Analytisc',
                    xaxis= { 'title': x },
                    yaxis= dict(title = y),
                    margin={ 'l': 40, 'b': 40, 't': 40, 'r': 10 },
                    hovermode='closest'
                )
            )

#CALLBACK TAB 4
@app.callback(
    Output(component_id='piegraph', component_property='figure'),
    [Input(component_id='groupplotpie', component_property='value')]
)
def callbackUpdatePieGraph(group):
    return dict(
                data=[
                    go.Pie(
                        labels=[Scatterlegend[group][val] for val in dfMobil[group].unique()],
                        values=[
                            len(dfMobil[dfMobil[group] == val])
                            for val in dfMobil[group].unique()
                        ]
                    )
                ],
                layout=go.Layout(
                    title='Pie Chart Cars Analytics',
                    margin={'l': 160, 'b': 40, 't': 40, 'r': 10}
                )
            )

listGoFunc = {
    'Bar': go.Bar,
    'Box': go.Box,
    'Violin': go.Violin
}

#CALLBACK TAB 5
@app.callback(
    Output(component_id='categorygraph', component_property='figure'),
    [Input(component_id='jenisplotcategory', component_property='value'),
    Input(component_id='xplotcategory', component_property='value'),
    Input(component_id='yplotcategory', component_property='value'),
    Input(component_id='statsplotcategory', component_property='value')]
)

def generateValuePlot(origin, x, y, stats = 'mean'):
    return {
        'x': {
            'Bar': dfMobil[dfMobil['origin'] == origin][x].unique(),
            'Box': dfMobil[dfMobil['origin'] == origin][x],
            'Violin': dfMobil[dfMobil['origin'] == origin][x]
        },
        'y': {
            'Bar': dfMobil[dfMobil['origin'] == origin].groupby(x)[y].describe()[stats],
            'Box': dfMobil[dfMobil['origin'] == origin][y],
            'Violin': dfMobil[dfMobil['origin'] == origin][y]
        }
    }

def callbackupdatecatgraph(jenisplot,x,y,stats):
    return dict(
        layout= go.Layout(
            title= '{} Plot Cars Analytics'.format(jenisplot),
            xaxis= { 'title': x },
            yaxis= dict(title=y),
            boxmode='group',
            violinmode='group'
        ),
        data=[
            listGoFunc[jenisplot](
                x=generateValuePlot('usa',x,y)['x'][jenisplot],
                y=generateValuePlot('usa',x,y,stats)['y'][jenisplot],
                name='USA'
            ),
            listGoFunc[jenisplot](
                x=generateValuePlot('japan',x,y)['x'][jenisplot],
                y=generateValuePlot('japan',x,y,stats)['y'][jenisplot],
                name='Japan'
            ),
            listGoFunc[jenisplot](
                x=generateValuePlot('europe',x,y)['x'][jenisplot],
                y=generateValuePlot('europe',x,y,stats)['y'][jenisplot],
                name='Europe'
            )
        ]
    )


#CALLBACK TAB 6
@app.callback(
    Output(component_id='histgraph', component_property='figure'),
    [Input(component_id='xplothist', component_property='value'),
    Input(component_id='hueplothist', component_property='value'),
    Input(component_id='stdplothist', component_property='value'),]
)
def update_hist_plot(x, hue, std):
    std = int(std)
    if(hue == 'All') :
        return dict(
                data=[
                    go.Histogram(
                        x=dfMobil[
                            (dfMobil[x] >= (dfMobil[x].mean() - (std * dfMobil[x].std())))
                            & (dfMobil[x] <= (dfMobil[x].mean() + (std * dfMobil[x].std())))
                        ][x],
                        name='Normal',
                        marker=dict(
                            color='green'
                        )
                    ),
                    go.Histogram(
                        x=dfMobil[
                            (dfMobil[x] < (dfMobil[x].mean() - (std * dfMobil[x].std())))
                            | (dfMobil[x] > (dfMobil[x].mean() + (std * dfMobil[x].std())))
                        ][x],
                        name='Not Normal',
                        marker=dict(
                            color='red'
                        )
                    )
                ],
                layout=go.Layout(
                    title='Histogram {} Stats Cars'.format(x),
                    xaxis=dict(title=x),
                    yaxis=dict(title='Count'),
                    height=450, width=1000
                )
            )
    subtitles = []
    for val in dfMobil[hue].unique() :
        dfSub = dfMobil[dfMobil[hue] == val]
        outlierCount = len(dfSub[
                        (dfSub[x] < (dfSub[x].mean() - (std * dfSub[x].std())))
                        | (dfSub[x] > (dfSub[x].mean() + (std * dfSub[x].std())))
                    ])
        subtitles.append(Scatterlegend[hue][val] + " ({}% outlier)".format(round(outlierCount/len(dfSub) * 100, 2)))

    fig = tools.make_subplots(
        rows=rowcolhist[hue]['row'], cols=rowcolhist[hue]['col'],
        subplot_titles=subtitles
    )
    uniqueData = dfMobil[hue].unique().reshape(rowcolhist[hue]['row'],rowcolhist[hue]['col'])
    index=1
    for r in range(1, rowcolhist[hue]['row']+1) :
        for c in range(1, rowcolhist[hue]['col']+1) :
            dfSub = dfMobil[dfMobil[hue] == uniqueData[r-1,c-1]]
            fig.append_trace(
                go.Histogram(
                    x=dfSub[
                        (dfSub[x] >= (dfSub[x].mean() - (std * dfSub[x].std())))
                        & (dfSub[x] <= (dfSub[x].mean() + (std * dfSub[x].std())))
                    ][x],
                    name='Normal {} {}'.format(hue,uniqueData[r-1,c-1]),
                    marker=dict(
                        color='green'
                    )
                ),r,c
            )
            fig.append_trace(
                go.Histogram(
                    x=dfSub[
                        (dfSub[x] < (dfSub[x].mean() - (std * dfSub[x].std())))
                        | (dfSub[x] > (dfSub[x].mean() + (std * dfSub[x].std())))
                    ][x],
                    name='Not Normal {} {}'.format(hue, uniqueData[r-1,c-1]),
                    marker=dict(
                        color='red'
                    )
                ),r,c
            )
            fig['layout']['xaxis'+str(index)].update(title=x.capitalize())
            fig['layout']['yaxis'+str(index)].update(title='Count')
            index += 1

    if(hue == 'Generation') :
        fig['layout'].update(height=700, width=1000,
                            title='Histogram {} Stats Pokemon'.format(x))
    else :
        fig['layout'].update(height=450, width=1000,
                            title='Histogram {} Stats Pokemon'.format(x))

    return fig




if __name__ == '__main__':
    app.run_server(debug=True)