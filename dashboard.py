#dash
import dash
from dash import dcc
from dash import html
from dash.dependencies import Output, Input

#plotting
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px

#data processing
import pandas as pd
import numpy as np

#modelling time series
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

#sklearn metrics
from sklearn.metrics import mean_absolute_percentage_error

#save models
import joblib

###### ADD CORRELATION TESTS AND MODEL RESULTS
###### FIX TRY AND IF IN TOP_CAUSES

#icd10 codes: https://raw.githubusercontent.com/k4m1113/ICD-10-CSV/master/codes.csv
#icd10 categories: https://raw.githubusercontent.com/k4m1113/ICD-10-CSV/master/categories.csv

#row1: pie chart (add others) + deaths TS
#row2: TS od code + decomposition

n_causes=50 #number of causes wanted

#importing data
df_sim = pd.read_csv('dashboard_sim.csv', index_col=0)
df_sim.DTOBITO = pd.to_datetime(df_sim.DTOBITO)
top_causes = df_sim.groupby(by=['CAUSABAS']).agg({'COUNT':'sum'}).sort_values(
    by='COUNT', ascending=False)[:n_causes+1]
#top_causes = top_causes.drop(top_causes.index[top_causes.index=='ALL'])
#code_to_name = pd.read_csv('data/icd/code_to_name.csv')

codes = pd.read_csv('codes.csv', header=None)
categories = pd.read_csv('categories.csv', header=None)
#code to name according to number
code_to_name = {}
for code in top_causes.index[:n_causes]:
    try:
        code_to_name[code] = codes[codes[1]==code][5].iloc[0]
    except:
        try:
            code_to_name[code] = codes[codes[2]==code][3].iloc[0]
        except:
            try:
                code_to_name[code] = categories[categories[0]==code][1].iloc[0]
            except:
                pass
code_to_name['ALL'] = 'All causes combined'

app = dash.Dash()
app.layout = html.Div(
    [html.Div(className='container-fluid', children=[html.H2('SIM - DATASUS'),
                                                      html.H4('ICD-10 code:'),
                                                      dcc.Input(id='icd10_code', value='', type='text')
                                                      ]),
     #html.Div(id)
     html.Div(className='row', children=[html.H5('Examples: '+', '.join([cause for cause in top_causes.index[:10]])+'. ALL for all causes.'),
                                         html.H6("Note that for some codes there's a change of pattern in 1996 due to\
                                          the translation from ICD9 to ICD10"),
                                        ]),
     html.Div(id='title'),
     html.Div(id='plot_codes_pie',
        style={'width': '40%', 'display':'inline-block'}),
     html.Div(id='plot_moving_averages',
        style={'width': '60%', 'display':'inline-block'}),
     html.Div(id='plot_decomposition',
        style={'width': '50%', 'display':'inline-block'}),
     html.Div(id='plot_seasonality',
        style={'width': '50%', 'display':'inline-block'}),
     #html.Div(id='plot_sarima_models'),
     #html.Div(id='plot_correlation_tests',
     #    style={'width': '50%', 'display':'inline-block'}),
     #html.Div(id='model_infos')

    ])


@app.callback(
    Output(component_id='title', component_property='children'),
    [Input(component_id='icd10_code', component_property='value')]
)
def update_plot(input_data):
    cause = input_data.upper()
    try:
        return html.H1(code_to_name[cause]+ ' ('+cause+')')
    except:
        return html.H4("Couldn't find code " + cause)

@app.callback(
    Output(component_id='plot_codes_pie', component_property='children'),
    [Input(component_id='icd10_code', component_property='value')]
)
def update_plot_codes_pie(input_data):
    cause = input_data.upper()

    try:

        if cause in top_causes.index:

            data_code_to_name = pd.DataFrame({'code':code_to_name.keys(),
                                              'name':code_to_name.values()})
            data = top_causes.drop(top_causes.index[top_causes.index=='ALL']).reset_index()
            data = data.merge(data_code_to_name, left_on='CAUSABAS', right_on='code')
            data = data.drop('code', axis=1)
            data.columns = ['Code', 'NumDeaths', 'Name']
            #other_causes = {'Code': 'OTHER',
                            #'NumDeaths': df_sim.COUNT.sum() - top_causes.COUNT.sum(),
                            #'Name':'Other causes'}
            #data = data.append(other_causes, ignore_index=True)


            #figure
            fig = px.pie(data_frame=data,
                        values='NumDeaths',
                        names='Code',
                        hover_name='Name')

            fig.update_layout(
                title_text='Pie chart of ICD-Codes <br><sup>'+"OBS: Considering only top-"+str(n_causes)+" causes"+"</sup>",
                title_font={'size':30})

            if cause=='ALL':
                fig.update_traces(marker=dict(line=dict(color='#000000', width=2)))
                return dcc.Graph(figure=fig, style={'width':'50vh', 'height':'60vh'})
            else:
                #slice to be pulled
                n = data[data.Code==cause].index.item()
                pull = np.zeros(n+1)
                pull[n] = 0.5
                fig.update_traces(marker=dict(line=dict(color='#000000', width=2)),
                                  pull=pull)

                return dcc.Graph(figure=fig, style={'width':'50vh', 'height':'60vh'})

        else: pass
    except: pass

@app.callback(
    Output(component_id='plot_moving_averages', component_property='children'),
    [Input(component_id='icd10_code', component_property='value'),]
)
def update_plot_moving_averages(input_data):
    cause = input_data.upper()

    try:

        if cause in top_causes.index:

            data = df_sim[df_sim.CAUSABAS==cause][['DTOBITO', 'COUNT']].set_index('DTOBITO')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data.COUNT, name='Original'))
            for step in np.arange(1,25,1):
                fig.add_trace(
                    go.Scatter(visible=False,
                               x=data.index,
                               y=data.COUNT.rolling(step).mean(), name='MA n='+str(step)))

            #start on 6
            fig.data[6].visible=True

            #create and add slider
            steps = []
            for i in range(len(fig.data)):
                step=dict(
                    method='update',
                    args=[{'visible':[True]+[False]*(len(fig.data)-1)},
                           {'title': 'Time Series and Moving Average n='+str(i)}])
                step['args'][0]['visible'][i] = True
                steps.append(step)

            sliders = [dict(
                active=6,
                currentvalue={"prefix": "n= "},
                #pad={"t": 50},
                steps=steps
            )]

            fig.update_layout(
                sliders=sliders,
                title_text='Time Series and Moving Average n=6',
                title_font={'size':30})

            return dcc.Graph(figure=fig, style={'width':'70vh', 'height':'60vh'})

        else: pass
    except: pass

@app.callback(
    Output(component_id='plot_decomposition', component_property='children'),
    [Input(component_id='icd10_code', component_property='value')]
)
def update_plot_decomposition(input_data):
    cause = input_data.upper()

    if cause in top_causes.index:

        data = df_sim[df_sim.CAUSABAS==cause][['DTOBITO', 'COUNT']].set_index('DTOBITO')
        dec = seasonal_decompose(data, model='additive', period=12)
        #dec.observed, dec.seasonal, dec.trend, dec.resid
        fig = make_subplots(rows=4, cols=1,
                            subplot_titles=('Observed', 'Trend', 'Seasonal',  'Residuals'))
        fig.add_trace(go.Scatter(
                      x = dec.observed.index,
                      y = dec.observed,
                      mode='lines',
                      name='observed'),
                     row=1, col=1)
        fig.add_trace(go.Scatter(
                      x = dec.trend.index,
                      y = dec.trend,
                      mode='lines',
                      name='trend'),
                     row=2, col=1)
        fig.add_trace(go.Scatter(
                      x = dec.seasonal.index,
                      y = dec.seasonal,
                      mode='lines',
                      name='seasonal',
                      line=dict(color='green', width=0.5)),
                     row=3, col=1)
        fig.add_trace(go.Scatter(
                      x = dec.resid.index,
                      y = dec.resid,
                      mode='markers',
                      name='resid'),
                     row=4, col=1)

        fig.add_hline(y=0, row=4, col=1, line_color='#000000')

        fig.update_layout(showlegend=False,
                          title_text='Decomposition',
                          title_font= {'size':30}
                        )

        return dcc.Graph(figure=fig, style={'width':'60vh', 'height':'80vh'})

    else:
        pass


@app.callback(
    Output(component_id='plot_seasonality', component_property='children'),
    [Input(component_id='icd10_code', component_property='value')]
)
def update_plot(input_data):
    cause = input_data.upper()

    if cause in top_causes.index:
        df = df_sim[df_sim.CAUSABAS==cause][['DTOBITO', 'COUNT']].set_index('DTOBITO')

        fig = make_subplots(rows=3, cols=1,
                    subplot_titles=('Mean deaths by year',
                                    'Mean deaths by month',
                                    'Mean deaths by weekday'))

        #yearly
        year_mean = df.groupby(df.index.year).mean()
        year_var = df.groupby(df.index.year).std()
        fig.add_trace(go.Bar(
                        x=year_mean.index,
                        y=year_mean.COUNT,
                        name='year',
                        error_y={'type':'data',
                                 'symmetric':False,
                                 'array':year_var.COUNT,
                                 'arrayminus':year_var.COUNT}),
                        row=1, col=1)

        #monthly
        month_mean = df.groupby(df.index.month).mean()
        month_var = df.groupby(df.index.month).std()
        fig.add_trace(go.Bar(
                        x=month_mean.index,
                        y=month_mean.COUNT,
                        name='month',
                        error_y={'type':'data',
                                 'symmetric':False,
                                 'array':month_var.COUNT,
                                 'arrayminus':month_var.COUNT}),
                        row=2, col=1)

        #weekday
        weekday_mean = df.groupby(df.index.weekday).mean()
        weekday_var = df.groupby(df.index.weekday).std()
        fig.add_trace(go.Bar(
                        x=weekday_mean.index,
                        y=weekday_mean.COUNT,
                        name='weekday',
                        error_y={'type':'data',
                                 'symmetric':False,
                                 'array':weekday_var.COUNT,
                                 'arrayminus':weekday_var.COUNT}),
                        row=3, col=1)

        fig.update_layout(showlegend=False,
                          title_text='Mean deaths by period',
                          title_font= {'size':30}
                        )

        return dcc.Graph(figure=fig, style={'width':'60vh', 'height':'80vh'})

    else:
        pass


# @app.callback(
#     Output(component_id='plot_sarima_models', component_property='children'),
#     [Input(component_id='icd10_code', component_property='value')]
# )
# def update_plot(input_data):
#     cause = input_data.upper()
#
#     if cause in top_causes.index:
#         #train and test split
#         df = df_sim[df_sim.CAUSABAS==cause].set_index('DTOBITO')['COUNT']
#         df_train = df[:-24]
#         df_test = df[-24:]
#
#         #model
#         arima_model = joblib.load('models/sim_sarima/'+cause+'_sarima.pkl')
#         model = SARIMAX(df_train, order=arima_model.get_params()['order'],
#                         seasonal_order=arima_model.get_params()['seasonal_order'])
#         results = model.fit(disp=0)
#
#         prediction = results.predict(start=1,
#                                     end=len(df_train)-1)
#         forecast_test = pd.Series(results.forecast(steps=12*2),
#                                  index= pd.date_range(start=df_test.index[0],
#                                  freq='MS',
#                                  periods=12*2))
#
#         forecast_extra = pd.Series(results.forecast(steps=12*(2+2)),
#                                  index= pd.date_range(start=df_test.index[-1] + pd.DateOffset(months=1),
#                                  freq='MS',
#                                  periods=12*2))
#
#         train_resids = results.resid[1:]
#         test_resids = forecast_test - df_test
#
#         train_resids_adfuller_pvalue = adfuller(train_resids)[1]
#         test_resids_adfuller_pvalue = adfuller(test_resids)[1]
#
#         train_score = mean_absolute_percentage_error(df_train[1:], prediction)
#         test_score = mean_absolute_percentage_error(df_test, forecast_test)
#
#         fig = make_subplots(rows=3, cols=1,
#                             subplot_titles=('Model Predictions and Forecast', 'Model Residuals'))
#
#         fig.add_trace(go.Scatter(x=df.index, y=df,
#                                  name='Time Series',
#                                  line_color='black'),
#                                  row=1, col=1)
#         fig.add_trace(go.Scatter(x=prediction.index, y=prediction,
#                                  name='Train Predictions',
#                                  line_color='blue'),
#                                  row=1, col=1)
#         fig.add_trace(go.Scatter(x=forecast_test.index, y=forecast_test,
#                                  name='Test Forecast',
#                                  line_color='red'),
#                                  row=1, col=1)
#         fig.add_trace(go.Scatter(x=forecast_extra.index, y=forecast_extra,
#                                  name='Real Forecast',
#                                  line_color='green'),
#                                  row=1, col=1)
#
#         fig.add_trace(go.Scatter(x=train_resids.index, y=train_resids, mode='markers',
#                                  name='Train Residuals',
#                                  line_color='blue'),
#                                  row=2, col=1)
#         fig.add_trace(go.Scatter(x=test_resids.index, y=test_resids, mode='markers',
#                                  name='Test Residuals',
#                                  line_color='red'),
#                                  row=2, col=1)
#         fig.add_hline(y=0, row=2, col=1, line_color='#000000')
#
#         fig.add_trace(go.Scatter(),row=3, col=1)
#
#         fig.add_annotation(text='<br>SARIMA order params: '+str(arima_model.get_params()['order'])+\
#                            '<br>SARIMA seasonal order params: '+str(arima_model.get_params()['seasonal_order'])+\
#                            '<br>Train MAPE error: '+str(train_score)+\
#                            '<br>Train Residuals adfuller pvalue: '+str(train_resids_adfuller_pvalue)+\
#                            '<br>Test MAPE error: '+str(test_score)+\
#                            '<br>Test Residuals adfuller pvalue: '+str(test_resids_adfuller_pvalue),
#                            showarrow=False, x=0.5, y=1,
#                            row=3, col=1)
#
#         fig.update_layout(
#             title_text='SARIMA model',
#             title_font={'size':30})
#
#         fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
#                           plot_bgcolor='rgba(0,0,0,0)',
#                           xaxis3=dict(visible=False, range=[0,1]),
#                           yaxis3=dict(visible=False, range=[0,1]))
#
#         return dcc.Graph(figure=fig, style={'width':'100vh', 'height':'70vh'})
#
#     else:
#         pass
#
# @app.callback(
#     Output(component_id='plot_correlation_tests', component_property='children'),
#     [Input(component_id='icd10_code', component_property='value')]
# )
# def update_plot(input_data):
#     cause = input_data.upper()
#
#     if cause in top_causes.index:
#         df = df_sim[df_sim.CAUSABAS==cause][['DTOBITO', 'COUNT']].set_index('DTOBITO')
#
#         #fig_acf = plot_acf(df, lags=48)
#         #fig_pacf = plot_pacf(df, lags=48)

if __name__ == '__main__':
    app.run_server(debug=True)
