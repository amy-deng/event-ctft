
import plotly.graph_objects as go
from plotly import graph_objs as go


import numpy as np

# Define random surface
N = 50
fig = go.Figure(data=[go.Mesh3d(x=(60*np.random.randn(N)),
                   y=(25*np.random.randn(N)),
                   z=(40*np.random.randn(N)),
                   opacity=0.5,
                   color='rgba(100,22,200,0.5)'
                  )])

# Different types of customized ticks
fig.update_layout(scene = dict(
                    xaxis = dict(
                        ticktext= ['TICKS','MESH','PLOTLY','PYTHON'],
                        tickvals= [0,50,75,-50]),
                    yaxis = dict(
                        nticks=5, tickfont=dict(
                            color='green',
                            size=12,
                            family='Old Standard TT, serif',),
                        ticksuffix='#'),
                    zaxis = dict(
                        nticks=4, ticks='outside',
                        tick0=0, tickwidth=4),),
                    width=700,
                    margin=dict(r=10, l=10, b=10, t=10)
                  )

fig.show()