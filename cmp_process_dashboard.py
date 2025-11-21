# -*- coding: utf-8 -*-
"""
@author: Edison Chukwuemeka
@date: 11/17/2025
File: sample_cmp_process_dashboard.py
PRODUCT: PyCharm
PROJECT: NioWave
"""
# Import the required Modules for the dashboard
import os
import dash
from dash import dcc, html, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc  # ADD THIS LINE
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Initialize with bootstrap for better styling
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "CMP Process Analytics Dashboard"


# Sample data generation (using our varied dataset)
def generate_cmp_data(seed=42):
    np.random.seed(seed)
    dates = pd.date_range('2024-01-01', periods=240, freq='H')

    df = pd.DataFrame({
        'timestamp': dates,
        'wafer_id': range(1, 241),
        'lot_id': np.repeat(range(1, 11), 24),
        'thickness': np.random.normal(725, 0.15, 240),
        'roughness': np.random.normal(1.3, 0.25, 240),
        'downforce_psi': np.random.normal(4.5, 0.3, 240),
        'platen_rpm': np.random.normal(120, 5, 240),
        'slurry_flow': np.random.normal(250, 15, 240),
        'pad_age_hours': np.random.randint(0, 48, 240),
        'defect_flag': np.random.choice([0, 1], 240, p=[0.977, 0.023]),
        'tool_id': np.random.choice(['Tool_A', 'Tool_B', 'Tool_C'], 240)
    })

    # Introduce variations
    df.loc[120:, 'thickness'] += 0.25
    df.loc[df['tool_id'] == 'Tool_B', 'roughness'] += 0.15
    df.loc[df['tool_id'] == 'Tool_C', 'slurry_flow'] *= 0.92
    df.loc[:, 'date'] = pd.to_datetime(df['timestamp']).dt.date

    # Add some outliers
    outlier_indices = [45, 89, 134, 187, 215]
    for idx in outlier_indices:
        df.loc[idx, 'roughness'] += np.random.uniform(0.8, 1.5)

    return df


# Initialize with a global dataframe
global global_df

global_df = generate_cmp_data()


# Specification limits
specs = {
    'thickness': {'usl': 725.5, 'lsl': 724.5, 'target': 725.0},
    'roughness': {'usl': 1.8, 'lsl': 0.5, 'target': 1.2},
    'downforce_psi': {'usl': 5.2, 'lsl': 3.8, 'target': 4.5},
    'platen_rpm': {'usl': 135, 'lsl': 105, 'target': 120},
    'slurry_flow': {'usl': 290, 'lsl': 210, 'target': 250}
}


# =============================================================================
# VISUALIZATION FUNCTIONS (Defined FIRST to avoid NameError)
# =============================================================================

def create_capability_gauge(value, title, max_value=2.0):
    """Create capability gauge chart"""

    # Determine color based on value
    if value >= 1.33:
        color = "green"
    elif value >= 1.0:
        color = "yellow"
    else:
        color = "red"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        delta={'reference': 1.33},
        gauge={
            'axis': {'range': [0, max_value], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 1.0], 'color': "lightcoral"},
                {'range': [1.0, 1.33], 'color': "lightyellow"},
                {'range': [1.33, max_value], 'color': "lightgreen"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 1.33}}
    ))

    fig.update_layout(height=250)
    return fig


def create_yield_gauge(yield_pct, title):
    """Create yield gauge chart"""

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=yield_pct,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 95], 'color': "lightcoral"},
                {'range': [95, 99], 'color': "lightyellow"},
                {'range': [99, 100], 'color': "lightgreen"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 99}}
    ))

    fig.update_layout(height=250)
    return fig


def create_capability_plot(df, parameter, spec):
    """Create capability analysis plot"""

    data = df[parameter]
    usl, lsl, target = spec['usl'], spec['lsl'], spec['target']
    mean_val = data.mean()
    std_val = data.std()

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Distribution with Specifications', 'Box Plot with Limits'))

    # Histogram
    fig.add_trace(
        go.Histogram(x=data, nbinsx=30, name='Distribution', opacity=0.7, marker_color='lightblue'),
        row=1, col=1
    )

    # Add specification lines
    for limit, name, color in [(usl, 'USL', 'red'), (lsl, 'LSL', 'red'),
                               (target, 'Target', 'green'), (mean_val, 'Mean', 'blue')]:
        fig.add_vline(x=limit, line_dash="dash", line_color=color,
                      annotation_text=name, row=1, col=1)

    # Box plot
    fig.add_trace(
        go.Box(y=data, name=parameter, boxpoints='outliers', marker_color='lightcoral'),
        row=1, col=2
    )

    # Add specification lines to box plot
    for limit, name, color in [(usl, 'USL', 'red'), (lsl, 'LSL', 'red'),
                               (target, 'Target', 'green')]:
        fig.add_hline(y=limit, line_dash="dash", line_color=color,
                      annotation_text=name, row=1, col=2)

    fig.update_layout(height=400, showlegend=False, title_text=f"Capability Analysis - {parameter.title()}")
    return fig


def create_hypothesis_gauge(p_value, alpha, title):
    """Create hypothesis testing gauge"""

    significant = p_value < alpha
    color = "green" if significant else "red"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{title}<br>{'SIGNIFICANT' if significant else 'NOT SIGNIFICANT'}",
               'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, alpha], 'color': "lightgreen"},
                {'range': [alpha, 1], 'color': "lightcoral"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': alpha}}
    ))

    fig.update_layout(height=250)
    return fig


def create_normality_gauge(p_value):
    """Create normality test gauge"""

    normal = p_value > 0.05
    color = "green" if normal else "red"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Normality Test<br>{'NORMAL' if normal else 'NON-NORMAL'}",
               'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': color},
            'steps': [
                {'range': [0.05, 1], 'color': "lightgreen"},
                {'range': [0, 0.05], 'color': "lightcoral"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.05}}
    ))

    fig.update_layout(height=250)
    return fig


def create_correlation_gauge(correlation):
    """Create correlation gauge"""

    abs_corr = abs(correlation)
    if abs_corr > 0.7:
        color = "red"
    elif abs_corr > 0.5:
        color = "orange"
    elif abs_corr > 0.3:
        color = "yellow"
    else:
        color = "green"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=correlation,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"Correlation with Defects<br>Strength: {['Weak', 'Moderate', 'Strong', 'Very Strong'][min(3, int(abs_corr * 4))]}",
            'font': {'size': 12}},
        gauge={
            'axis': {'range': [-1, 1]},
            'bar': {'color': color},
            'steps': [
                {'range': [-1, -0.7], 'color': "lightcoral"},
                {'range': [-0.7, -0.5], 'color': "lightyellow"},
                {'range': [-0.5, 0.5], 'color': "lightgreen"},
                {'range': [0.5, 0.7], 'color': "lightyellow"},
                {'range': [0.7, 1], 'color': "lightcoral"}],
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 0.75,
                'value': 0}}
    ))

    fig.update_layout(height=250)
    return fig


def create_effect_gauge(effect, title):
    """Create DOE effect gauge"""

    abs_effect = abs(effect)
    if abs_effect > 0.1:
        color = "red"
    elif abs_effect > 0.05:
        color = "orange"
    else:
        color = "green"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=effect,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 14}},
        gauge={
            'axis': {'range': [-0.2, 0.2]},
            'bar': {'color': color},
            'steps': [
                {'range': [-0.2, -0.1], 'color': "lightcoral"},
                {'range': [-0.1, -0.05], 'color': "lightyellow"},
                {'range': [-0.05, 0.05], 'color': "lightgreen"},
                {'range': [0.05, 0.1], 'color': "lightyellow"},
                {'range': [0.1, 0.2], 'color': "lightcoral"}],
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 0.75,
                'value': 0}}
    ))

    fig.update_layout(height=250)
    return fig


def create_control_charts(df):
    """Create control charts for process overview"""

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Thickness Control Chart', 'Roughness Control Chart',
                                        'Downforce Control Chart', 'Platen RPM Control Chart'))

    parameters = ['thickness', 'roughness', 'downforce_psi', 'platen_rpm']
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    # Add traces to the main figure, managing legend properties in the loop
    traces_to_show_legend = set()  # Track which names we have already shown a legend for

    for param, (row, col) in zip(parameters, positions):
        data = df[param]
        mean_val = data.mean()
        std_val = data.std()
        ucl = mean_val + 3 * std_val
        lcl = mean_val - 3 * std_val

        fig_scatter = px.scatter(data_frame=df, x='timestamp', y=param,color='tool_id', symbol='tool_id',)
        fig_line = px.line(data_frame=df, x='timestamp', y=param,color='tool_id',)
        # Add data trace
        fig_data = fig_scatter.data + fig_line.data

        for trace in fig_data:
            # Ensure all traces for the same 'tool_id' belong to the same legend group
            trace.update(legendgroup=trace.name)

            # Decide if this specific trace should have its legend shown
            if trace.name not in traces_to_show_legend:
                # Show legend for the first trace of this group we encounter
                trace.update(showlegend=True)
                traces_to_show_legend.add(trace.name)
            else:
                # Hide legend for subsequent traces in the same group
                trace.update(showlegend=False)

            fig.add_trace(
                #go.Scatter(x=df.index, y=data, mode='lines+markers', name=param,
                #           line=dict(color='blue'), marker=dict(size=3)),
                trace,
                row=row, col=col
            )

        # Add control limits
        for limit, name, color, style in [(ucl, 'UCL', 'red', 'dash'),
                                          (lcl, 'LCL', 'red', 'dash'),
                                          (mean_val, 'CL', 'green', 'solid')]:
            fig.add_hline(y=limit, line_dash=style, line_color=color,
                          annotation_text=name, row=row, col=col)

    fig.update_layout(height=600, showlegend=True, title_text="Process Control Charts")
    return fig


def create_tool_performance_chart(df, parameter):
    """Create tool performance comparison chart"""

    fig = px.box(df, x='tool_id', y=parameter, color='tool_id',
                 title=f'{parameter.title()} Distribution by Tool')
    fig.update_layout(showlegend=False)
    return fig


def create_defect_trend_chart(df):
    """Create defect trend chart"""

    daily_defects = df.groupby(df['timestamp'].dt.date)['defect_flag'].mean() * 100

    fig = px.line(x=daily_defects.index, y=daily_defects.values,
                  title='Daily Defect Rate Trend',
                  labels={'x': 'Date', 'y': 'Defect Rate (%)'})
    fig.add_hline(y=1.5, line_dash="dash", line_color="red", annotation_text="Target")
    return fig


def create_defect_by_tool_chart(df):
    """Create defect rate by tool chart"""

    tool_defects = df.groupby('tool_id')['defect_flag'].mean() * 100

    fig = px.bar(x=tool_defects.index, y=tool_defects.values,
                 title='Defect Rate by Tool',
                 labels={'x': 'Tool ID', 'y': 'Defect Rate (%)'},
                 color=tool_defects.values,
                 color_continuous_scale='RdYlGn_r')
    return fig


def create_correlation_heatmap(df):
    """Create correlation heatmap"""

    numeric_columns = ['thickness', 'roughness', 'downforce_psi', 'platen_rpm', 'slurry_flow', 'defect_flag']
    corr_matrix = df[numeric_columns].corr()

    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                    color_continuous_scale="RdBu_r", title="Parameter Correlation Matrix")
    return fig


def create_hypothesis_visualizations(df, parameter):
    """Create hypothesis testing visualizations"""

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(f'{parameter} by Defect Status', f'{parameter} by Tool'))

    # Defect status comparison
    fig.add_trace(
        go.Box(x=df['defect_flag'].astype(str), y=df[parameter],
               name='Defect Status', boxpoints='outliers'),
        row=1, col=1
    )

    # Tool comparison
    fig.add_trace(
        go.Box(x=df['tool_id'], y=df[parameter],
               name='Tool', boxpoints='outliers'),
        row=1, col=2
    )

    fig.update_layout(height=400, title_text=f"Hypothesis Testing Visualizations - {parameter.title()}")
    return fig


def create_doe_plots(doe_results):
    """Create DOE analysis plots"""

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Main Effects - Roughness', 'Main Effects - Removal Rate',
                                        'Main Effects - Uniformity', 'Interaction Plot'))

    responses = ['Roughness', 'Removal_Rate', 'Uniformity']
    factors = ['A', 'B', 'C']

    # Map factors to names
    factor_names = {'A': 'Downforce', 'B': 'Platen Speed', 'C': 'Slurry Flow'}

    # Main effects plots
    for i, response in enumerate(responses):
        row = 1 if i < 2 else 2
        col = 1 if i % 2 == 0 else 2

        for factor in factors:
            low_val = doe_results['design_matrix'][doe_results['design_matrix'][factor] == -1][response].mean()
            high_val = doe_results['design_matrix'][doe_results['design_matrix'][factor] == 1][response].mean()

            fig.add_trace(
                go.Scatter(x=[f'{factor_names[factor]}(-)', f'{factor_names[factor]}(+)'],
                           y=[low_val, high_val], mode='lines+markers', name=f'Factor {factor}'),
                row=row, col=col
            )

    # Interaction plot (simplified)
    interaction_data = []
    for a_level in [-1, 1]:
        for b_level in [-1, 1]:
            mask = (doe_results['design_matrix']['A'] == a_level) & (doe_results['design_matrix']['B'] == b_level)
            roughness_mean = doe_results['design_matrix'][mask]['Roughness'].mean()
            interaction_data.append({
                'A': 'Low' if a_level == -1 else 'High',
                'B': 'Low' if b_level == -1 else 'High',
                'Roughness': roughness_mean
            })

    interaction_df = pd.DataFrame(interaction_data)

    # Create heatmap for interaction
    pivot_data = interaction_df.pivot(index='A', columns='B', values='Roughness')

    fig.add_trace(
        go.Heatmap(z=pivot_data.values,
                   x=['Low', 'High'],
                   y=['Low', 'High'],
                   colorscale='Viridis',
                   showscale=True,
                   colorbar=dict(title="Roughness")),
        row=2, col=2
    )

    fig.update_layout(height=600, title_text="Design of Experiments Analysis")
    return fig


def create_capability_table():
    """Create capability summary table"""

    table_data = []
    for param, spec in specs.items():
        data = global_df[param]
        mean_val = data.mean()
        std_val = data.std()
        usl, lsl = spec['usl'], spec['lsl']

        cpk = min((usl - mean_val) / (3 * std_val), (mean_val - lsl) / (3 * std_val)) if std_val > 0 else 0
        cp = (usl - lsl) / (6 * std_val) if std_val > 0 else 0

        # Yield calculation
        z_usl = (usl - mean_val) / std_val if std_val > 0 else 0
        z_lsl = (mean_val - lsl) / std_val if std_val > 0 else 0
        p_defect = stats.norm.cdf(-z_usl) + stats.norm.cdf(-z_lsl)
        yield_pct = (1 - p_defect) * 100
        ppm = p_defect * 1e6

        # Status
        if cpk >= 1.33:
            status = "✅ Excellent"
        elif cpk >= 1.0:
            status = "⚠️ Marginal"
        else:
            status = "❌ Poor"

        table_data.append({
            'Parameter': param.title(),
            'Cpk': round(cpk, 3),
            'Cp': round(cp, 3),
            'Yield': round(yield_pct, 2),
            'PPM': int(ppm),
            'Status': status
        })

    return table_data


def perform_hypothesis_tests(df, parameter, alpha):
    """Perform hypothesis tests and return results"""

    # Defective vs Non-defective T-test
    defective = df[df['defect_flag'] == 1][parameter]
    non_defective = df[df['defect_flag'] == 0][parameter]

    if len(defective) > 1 and len(non_defective) > 1:
        t_stat, p_value_t = stats.ttest_ind(defective, non_defective, equal_var=False, nan_policy='omit')
    else:
        t_stat, p_value_t = 0, 1.0

    # Tool ANOVA
    tools = df['tool_id'].unique()
    tool_groups = [df[df['tool_id'] == tool][parameter] for tool in tools]

    # Check if we have enough data for ANOVA
    if all(len(group) > 1 for group in tool_groups):
        f_stat, p_value_anova = stats.f_oneway(*tool_groups)
    else:
        f_stat, p_value_anova = 0, 1.0

    # Normality test
    if len(df[parameter]) > 3 and len(df[parameter]) < 5000:
        w_stat, p_value_norm = stats.shapiro(df[parameter])
    else:
        w_stat, p_value_norm = 0, 1.0

    # Correlation with defects
    if len(df[parameter]) > 1:
        corr_coef, p_value_corr = stats.pearsonr(df[parameter], df['defect_flag'])
    else:
        corr_coef, p_value_corr = 0, 1.0

    return {
        'defect_t_test': {
            't_stat': t_stat, 'p_value': p_value_t,
            'defective_mean': defective.mean() if len(defective) > 0 else 0,
            'non_defective_mean': non_defective.mean() if len(non_defective) > 0 else 0,
            'conclusion': 'SIGNIFICANT' if p_value_t < alpha else 'NOT SIGNIFICANT'
        },
        'tool_anova': {
            'f_stat': f_stat, 'p_value': p_value_anova,
            'conclusion': 'SIGNIFICANT' if p_value_anova < alpha else 'NOT SIGNIFICANT'
        },
        'normality': {
            'w_stat': w_stat, 'p_value': p_value_norm,
            'conclusion': 'NORMAL' if p_value_norm > 0.05 else 'NON-NORMAL'
        },
        'defect_correlation': {
            'correlation': corr_coef, 'p_value': p_value_corr,
            'conclusion': 'SIGNIFICANT' if p_value_corr < alpha else 'NOT SIGNIFICANT'
        }
    }


def perform_doe_analysis():
    """Perform DOE analysis and return results"""

    # Define factors and levels
    factors = {
        'A': {'name': 'Downforce', 'low': 4.0, 'high': 5.0, 'center': 4.5},
        'B': {'name': 'Platen Speed', 'low': 110, 'high': 130, 'center': 120},
        'C': {'name': 'Slurry Flow', 'low': 230, 'high': 270, 'center': 250}
    }

    # Generate full factorial design
    from itertools import product
    design_points = list(product([-1, 1], repeat=3))
    design_df = pd.DataFrame(design_points, columns=['A', 'B', 'C'])

    # Add center points
    center_points = pd.DataFrame([{'A': 0, 'B': 0, 'C': 0} for _ in range(4)])
    design_df = pd.concat([design_df, center_points], ignore_index=True)

    # Convert to actual values
    design_df['Downforce_actual'] = design_df['A'].apply(
        lambda x: factors['A']['low'] if x == -1 else
        (factors['A']['high'] if x == 1 else factors['A']['center'])
    )
    design_df['PlatenSpeed_actual'] = design_df['B'].apply(
        lambda x: factors['B']['low'] if x == -1 else
        (factors['B']['high'] if x == 1 else factors['B']['center'])
    )
    design_df['SlurryFlow_actual'] = design_df['C'].apply(
        lambda x: factors['C']['low'] if x == -1 else
        (factors['C']['high'] if x == 1 else factors['C']['center'])
    )

    # Simulate responses
    np.random.seed(42)
    design_df['Roughness'] = (
            1.3 + 0.12 * design_df['A'] + 0.06 * design_df['B'] - 0.09 * design_df['C'] +
            np.random.normal(0, 0.05, len(design_df))
    )
    design_df['Removal_Rate'] = (
            150 + 18 * design_df['A'] + 9 * design_df['B'] + 6 * design_df['C'] +
            np.random.normal(0, 3, len(design_df))
    )
    design_df['Uniformity'] = (
            0.95 - 0.10 * design_df['A'] - 0.05 * design_df['B'] + 0.08 * design_df['C'] +
            np.random.normal(0, 0.03, len(design_df))
    )

    # Calculate main effects
    main_effects = {}
    for factor in ['A', 'B', 'C']:
        high_mean = design_df[design_df[factor] == 1]['Roughness'].mean()
        low_mean = design_df[design_df[factor] == -1]['Roughness'].mean()
        main_effects[factor] = (high_mean - low_mean) / 2

    return {
        'design_matrix': design_df.round(4),
        'main_effects': main_effects,
        'optimal_roughness': design_df['Roughness'].min(),
        'optimal_removal_rate': design_df['Removal_Rate'].max(),
        'optimal_uniformity': design_df['Uniformity'].max(),
        'optimal_settings': {
            'roughness': {'downforce': '4.0 psi', 'platen_speed': '130 rpm', 'slurry_flow': '270 ml/min'},
            'removal_rate': {'downforce': '5.0 psi', 'platen_speed': '130 rpm', 'slurry_flow': '270 ml/min'},
            'balanced': {'downforce': '4.5 psi', 'platen_speed': '120 rpm', 'slurry_flow': '260 ml/min'}
        }
    }


# =============================================================================
# DASH APP LAYOUT
# =============================================================================
# Sidebar style
column_style = {'width': '40%', 'height': '60%'}
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 10,
    "width": "15%",
    "padding": "3em 2em",
    'backgroundColor': '#2fad98',
    'color': '#161966', 'height': '100vh', 'overflow-y': 'auto',
    'overflow-x': 'scroll',
    'margin-bottom': '2rem',
    'display': 'inline-block',
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-top": "10rem",
    "margin-right": "2rem",
    'left': 20,
    "padding": "2rem",
    'position': 'float',
    'overflow-y': 'auto',
    'overflow-x': 'auto',
    'display': 'flex',
    'margin-bottom': '10rem',
}

SIDEBAR_LABEL_STYLE = {'fontWeight': 'bold', 'marginTop': '1rem'}
SIDEBAR_CONTENT_STYLE = {'fontSize': '14px', 'fontWeight': 'bold', 'marginTop': '1rem'}
TAB_LABEL_STYLE = {'fontWeight': 'bold', 'fontSize': '20px', 'color': '#161966', 'backgroundColor': '#5e6c73', }
selected_tab = {'backgroundColor': '#a0a0a0', 'color': 'purple'}

app.layout = html.Div([
    # Welcome Modal
    dbc.Modal([
        dbc.ModalHeader("🔬 Welcome to CMP Process Analytics Dashboard"),
        dbc.ModalBody([
            html.H4("Your Analytics Journey:", style={'color': '#2c3e50'}),
            html.Ol([
                html.Li(html.Strong("Process Overview: "),
                        "Check real-time process health and monitor production line"),
                html.Li(html.Strong("Process Capability: "), "Verify specification compliance and calculate yield"),
                html.Li(html.Strong("Hypothesis Testing: "), "Investigate root causes with statistical evidence"),
                html.Li(html.Strong("Design of Experiments: "), "Optimize process parameters for best results")
            ]),
            html.Hr(),
            html.P("Each tab contains expandable guides explaining its purpose and how to use it effectively.",
                   style={'fontStyle': 'italic', 'color': '#7f8c8d'}),
        ]),
        dbc.ModalFooter(
            dbc.Button("Start Analyzing", id="close-welcome", className="ml-auto", color="primary")
        ),
    ], id="welcome-modal", is_open=True, size="lg"),

    html.Div([
        html.H1("🔬 CMP Process Analytics Dashboard",
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 20}),
        html.P("Real-time Process Capability, Hypothesis Testing & DOE Analysis",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': 30}),
    ]),

    # Sidebar
    html.Div([
        html.Div([
            # Tab Information Panel
            html.Div([
                html.H4("📋 Tab Guide", style=SIDEBAR_LABEL_STYLE), #{'color': 'white', 'marginBottom': '10px'}),
                html.Div(id="tab-info-panel", style={
                    'backgroundColor': 'rgba(255,255,255,0.1)',
                    'padding': '10px',
                    'borderRadius': '5px',
                    'fontSize': '14px',
                    'color': 'darkblue',
                    'minHeight': '120px'
                })
            ], style={'marginBottom': '30px'}),

            # Parameter selection
            html.H4("Analysis Controls", style=SIDEBAR_LABEL_STYLE),
            html.Label("Select Parameter:", style=SIDEBAR_LABEL_STYLE),
            dcc.Dropdown(
                id='parameter-selector',
                options=[
                    {'label': 'Thickness', 'value': 'thickness'},
                    {'label': 'Roughness', 'value': 'roughness'},
                    {'label': 'Downforce', 'value': 'downforce_psi'},
                    {'label': 'Platen RPM', 'value': 'platen_rpm'},
                    {'label': 'Slurry Flow', 'value': 'slurry_flow'}
                ],
                value='thickness',
                clearable=False
            ),

            html.Label("Confidence Level:", style=SIDEBAR_LABEL_STYLE),
            dcc.Dropdown(
                id='confidence-level',
                options=[
                    {'label': '90%', 'value': 0.90},
                    {'label': '95%', 'value': 0.95},
                    {'label': '99%', 'value': 0.99}
                ],
                value=0.95,
                clearable=False
            ),

            html.Button('🔄 Refresh Analysis', id='refresh-button', n_clicks=0,
                        style={**SIDEBAR_LABEL_STYLE, 'marginTop': '20px', 'width': '100%'}),

        ], style=SIDEBAR_LABEL_STYLE),
    ], style=SIDEBAR_STYLE),

    # Main content area
    html.Div([
        # Real-time metrics row
        html.Div([
            dcc.Tabs(id="main-tabs", value='overview-tab', children=[
                dcc.Tab(label='📊 Process Overview', value='overview-tab',
                        style=TAB_LABEL_STYLE, selected_style=selected_tab),
                dcc.Tab(label='🎯 Process Capability', value='capability-tab',
                        style=TAB_LABEL_STYLE, selected_style=selected_tab),
                dcc.Tab(label='🔍 Hypothesis Testing', value='hypothesis-tab',
                        style=TAB_LABEL_STYLE, selected_style=selected_tab),
                dcc.Tab(label='⚗️ Design of Experiments', value='doe-tab',
                        style=TAB_LABEL_STYLE, selected_style=selected_tab),
            ]),
            html.Hr(),
            html.Div([
                html.H4("Overall Defect Rate", style={'textAlign': 'center'}),
                html.H2(id="defect-rate-metric",  # Add an ID for dynamic updating
                        style={'textAlign': 'center', 'color': '#e74c3c'}),
            ], className='metric-card', style={'width': '15%', 'display': 'inline-block', 'margin': '5px'}),

            html.Div([
                html.H4("Wafers Processed", style={'textAlign': 'center'}),
                html.H2(id="wafers-processed-metric",
                        style={'textAlign': 'center', 'color': '#3498db'}),
            ], className='metric-card', style={'width': '15%', 'display': 'inline-block', 'margin': '5px'}),

            html.Div([
                html.H4("Tools Active", style={'textAlign': 'center'}),
                html.H2(id="tools-active-metric",
                        style={'textAlign': 'center', 'color': '#27ae60'}),
            ], className='metric-card', style={'width': '15%', 'display': 'inline-block', 'margin': '5px'}),

            html.Div([
                html.H4("Lots Completed", style={'textAlign': 'center'}),
                html.H2(id="lots-completed-metric",
                        style={'textAlign': 'center', 'color': '#f39c12'}),
            ], className='metric-card', style={'width': '15%', 'display': 'inline-block', 'margin': '5px'}),
        ], style={'marginBottom': 30, 'marginLeft': '270px'}),

        # Tab content
        html.Div(id='tab-content', style={'marginLeft': '270px', 'padding': '20px'})

    ], style={'marginLeft': '25px'})
])


# =============================================================================
# CALLBACKS
# =============================================================================

@app.callback(
    Output('tab-content', 'children'),
    [Input('main-tabs', 'value'),
     Input('parameter-selector', 'value'),
     Input('confidence-level', 'value'),
     Input('refresh-button', 'n_clicks')]
)
def render_tab_content(selected_tab, selected_param, confidence_level, n_clicks):
    """Render content based on selected tab"""

    if selected_tab == 'capability-tab':
        return render_capability_tab(selected_param)
    elif selected_tab == 'hypothesis-tab':
        return render_hypothesis_tab(selected_param, confidence_level)
    elif selected_tab == 'doe-tab':
        return render_doe_tab()
    elif selected_tab == 'overview-tab':
        return render_overview_tab()
    else:
        return html.Div("Select a tab to view analysis")


def render_capability_tab(parameter):
    """Render process capability analysis tab"""

    spec = specs[parameter]
    data = global_df[parameter]

    # Calculate capability indices
    mean_val = data.mean()
    std_val = data.std()
    usl, lsl, target = spec['usl'], spec['lsl'], spec['target']

    cp = (usl - lsl) / (6 * std_val) if std_val > 0 else 0
    cpk = min((usl - mean_val) / (3 * std_val), (mean_val - lsl) / (3 * std_val)) if std_val > 0 else 0
    ppk = min((usl - mean_val) / (3 * data.std(ddof=0)), (mean_val - lsl) / (3 * data.std(ddof=0)))

    # Yield calculations
    z_usl = (usl - mean_val) / std_val if std_val > 0 else 0
    z_lsl = (mean_val - lsl) / std_val if std_val > 0 else 0
    p_defect = stats.norm.cdf(-z_usl) + stats.norm.cdf(-z_lsl)
    yield_pct = (1 - p_defect) * 100

    return html.Div([
        html.H2(f"🎯 Process Capability Analysis - {parameter.title()}"),

        # STORYTELLING SECTION
        html.Div([
            html.Details([
                html.Summary("📈 Understanding Process Capability (Click for Story & Summary)",
                             style={'cursor': 'pointer', 'color': '#2c3e50', 'fontWeight': 'bold', 'fontSize': '18px'}),
                html.Div([
                    html.Hr(),
                    html.P( html.Strong("📖 The Story:"), style={'fontWeight': 'bold', 'color': '#34495e'}),
                    html.P(
                        "'Now that you've confirmed the process is stable, the next question is: Is it capable of meeting customer specifications? This tab answers that definitively. It's like putting the process under a microscope. You can select any parameter and see its distribution relative to the specification limits. The Cpk/Cp gauges give you an instant, intuitive read on capability, while the yield calculation translates this into business impact (PPM). This is your go-to tab for qualifying a tool for production or validating process changes.'"),

                    html.Hr(),
                    html.P( html.Strong("📋 Summary:"), style={'fontWeight': 'bold', 'color': '#34495e'}),
                    html.Ul([
                        html.Li([html.Strong("Purpose: "),
                                "Quantify how well the process meets specification limits (Voice of the Customer vs. Voice of the Process)."]),
                        html.Li([html.Strong("Key Metrics: "), "Cpk, Cp, Yield %, PPM, Sigma Level"]),
                        html.Li([html.Strong("Best For: "),
                                "Quality engineers, process engineers, and product engineers."]),
                        html.Li([html.Strong("Key Question: "), "'Can we consistently meet customer requirements?'"])
                    ])
                ], style={'padding': '20px', 'backgroundColor': '#e8f4f8', 'borderRadius': '10px',
                          'borderLeft': '4px solid #e74c3c'})
            ])
        ], style={'marginBottom': '30px'}),

        html.Hr(),

        # QUICK SUMMARY CARDS
        html.Div([
            html.Div([
                html.H4("🎯 Purpose", style={'margin': '0', 'color': '#2c3e50'}),
                html.P("Specification Compliance", style={'margin': '5px 0', 'fontSize': '14px', 'color': '#7f8c8d'})
            ], style={'width': '24%', 'display': 'inline-block', 'padding': '15px', 'textAlign': 'center',
                      'backgroundColor': '#fef9e7', 'margin': '5px', 'borderRadius': '8px'}),

            html.Div([
                html.H4("📊 Key Metrics", style={'margin': '0', 'color': '#2c3e50'}),
                html.P("Cpk, Yield, PPM", style={'margin': '5px 0', 'fontSize': '14px', 'color': '#7f8c8d'})
            ], style={'width': '24%', 'display': 'inline-block', 'padding': '15px', 'textAlign': 'center',
                      'backgroundColor': '#fef9e7', 'margin': '5px', 'borderRadius': '8px'}),

            html.Div([
                html.H4("👥 Best For", style={'margin': '0', 'color': '#2c3e50'}),
                html.P("Quality Engineers", style={'margin': '5px 0', 'fontSize': '14px', 'color': '#7f8c8d'})
            ], style={'width': '24%', 'display': 'inline-block', 'padding': '15px', 'textAlign': 'center',
                      'backgroundColor': '#fef9e7', 'margin': '5px', 'borderRadius': '8px'}),

            html.Div([
                html.H4("⏱️ Frequency", style={'margin': '0', 'color': '#2c3e50'}),
                html.P("Weekly/Monthly Review", style={'margin': '5px 0', 'fontSize': '14px', 'color': '#7f8c8d'})
            ], style={'width': '24%', 'display': 'inline-block', 'padding': '15px', 'textAlign': 'center',
                      'backgroundColor': '#fef9e7', 'margin': '5px', 'borderRadius': '8px'}),
        ], style={'marginBottom': '30px'}),
        html.Hr(),
        # Capability gauges row
        html.Div([
            html.Div([
                dcc.Graph(
                    figure=create_capability_gauge(cpk, f"Cpk - {parameter}", 2.0),
                    style={'height': '300px'}
                )
            ], style={'width': '32%', 'display': 'inline-block', 'padding': '10px'}),

            html.Div([
                dcc.Graph(
                    figure=create_capability_gauge(cp, f"Cp - {parameter}", 2.0),
                    style={'height': '300px'}
                )
            ], style={'width': '32%', 'display': 'inline-block', 'padding': '10px'}),

            html.Div([
                dcc.Graph(
                    figure=create_yield_gauge(yield_pct, f"Yield - {parameter}"),
                    style={'height': '300px'}
                )
            ], style={'width': '32%', 'display': 'inline-block', 'padding': '10px'}),
        ]),

        # Statistical summary
        html.Div([
            html.H3("Statistical Summary"),
            html.Div([
                html.Div([
                    html.P(f"Mean: {mean_val:.4f}"),
                    html.P(f"Standard Deviation: {std_val:.4f}"),
                    html.P(f"USL: {usl}, LSL: {lsl}"),
                ], style={'width': '32%', 'display': 'inline-block', 'padding': '10px'}),

                html.Div([
                    html.P(f"Ppk: {ppk:.3f}"),
                    html.P(f"Defect Rate: {p_defect * 100:.4f}%"),
                    html.P(f"PPM: {p_defect * 1e6:.0f}"),
                ], style={'width': '32%', 'display': 'inline-block', 'padding': '10px'}),

                html.Div([
                    html.P(f"Sigma Level: {abs(stats.norm.ppf(p_defect / 2)):.2f}σ"),
                    html.P(f"Within Spec: {((data >= lsl) & (data <= usl)).mean() * 100:.1f}%"),
                    html.P(f"Sample Size: {len(data)}"),
                ], style={'width': '32%', 'display': 'inline-block', 'padding': '10px'}),
            ]),
        ], style={'marginTop': 20, 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),

        # Distribution plots
        html.Div([
            dcc.Graph(
                figure=create_capability_plot(global_df, parameter, spec)
            )
        ], style={'marginTop': 20}),

        # All parameters capability table
        html.Div([
            html.H3("All Parameters Capability Summary"),
            dash_table.DataTable(
                id='capability-table',
                columns=[
                    {"name": "Parameter", "id": "Parameter"},
                    {"name": "Cpk", "id": "Cpk"},
                    {"name": "Cp", "id": "Cp"},
                    {"name": "Yield %", "id": "Yield"},
                    {"name": "PPM", "id": "PPM"},
                    {"name": "Status", "id": "Status"}
                ],
                data=create_capability_table(),
                style_cell={'textAlign': 'center'},
                style_header={'backgroundColor': '#2c3e50', 'color': 'white', 'fontWeight': 'bold'}
            )
        ], style={'marginTop': 30})
    ])


def render_hypothesis_tab(parameter, confidence_level):
    """Render hypothesis testing tab"""

    alpha = 1 - confidence_level
    test_results = perform_hypothesis_tests(global_df, parameter, alpha)

    return html.Div([
        html.H2("Hypothesis Testing Analysis"),
        html.Hr(),
        # STORYTELLING SECTION
        html.Div([
            html.Details([
                html.Summary("🕵️‍♂️ Process Data Detective (Click for Story & Summary)",
                             style={'cursor': 'pointer', 'color': '#2c3e50', 'fontWeight': 'bold', 'fontSize': '18px'}),
                html.Div([
                    html.Hr(),
                    html.P(["📖 ", html.Strong("The Story:")], style={'fontWeight': 'bold', 'color': '#34495e'}),
                    html.P([
                        " Suppose you've noticed Tool B has a higher defect rate in the Overview tab, and your gut says ",
                        "defective wafers might be thicker. But in manufacturing, we need more than intuition—we ",
                        "need evidence. Welcome to your forensic lab. This tab provides the statistical proof to ",
                        "support or refute your hunches. Is Tool B truly different from the others? Are defective ",
                        "wafers statistically different in thickness? By running T-tests and ANOVAs, we move from ",
                        "correlation to causation. Think of yourself as a data detective: the p-values are your evidence,",
                        " the confidence level is your standard of proof, and these gauges are your lie detectors ",
                        "telling you which leads are worth pursuing."]),

                    html.Hr(),
                    html.P(["🔍 " , html.Strong("The Investigation Framework:")],
                           style={'fontWeight': 'bold', 'color': '#34495e'}),
                    html.Ol([
                        html.Li([html.Strong("Defective vs Non-Defective T-Test: "),
                                "Are bad wafers fundamentally different from good ones?"]),
                        html.Li([html.Strong("Tool ANOVA: "),
                                "Do our tools actually perform differently, or is it just random variation?"]),
                        html.Li([html.Strong("Normality Check: "),
                                "Is our data well-behaved enough for these statistical tests?"]),
                        html.Li([html.Strong("Correlation Analysis: "),
                                "Which parameters move together with defect rates?"])
                    ]),

                    html.Hr(),
                    html.P(["📋 " , html.Strong("Summary:")], style={'fontWeight': 'bold', 'color': '#34495e'}),
                    html.Ul([
                        html.Li([html.Strong("Purpose: "),
                                "Use statistical tests to objectively identify significant differences and relationships in the data."]),
                        html.Li([html.Strong("Key Features: "),
                                "Hypothesis gauges, ANOVA testing, correlation analysis, normality validation"]),
                        html.Li([html.Strong("Best For: "),
                                "Process engineers, data scientists, and advanced users conducting root-cause analysis."]),
                        html.Li([html.Strong("Key Question: "),
                                "'Why are we seeing these defects or differences, and is it statistically significant?'"])
                    ]),

                    html.Hr(),
                    html.P(["💡 " , html.Strong("Pro Tip:")], style={'fontWeight': 'bold', 'color': '#34495e'}),
                    html.P([
                        "A p-value below your confidence level (e.g., < 0.05 for 95% confidence) means the difference",
                        " is statistically significant. Green gauges mean 'strong evidence', red means 'not enough proof'."],
                        style={'fontStyle': 'italic', 'backgroundColor': '#fff3cd', 'padding': '10px',
                               'borderRadius': '5px'})
                ], style={'padding': '20px', 'backgroundColor': '#e8f4f8', 'borderRadius': '10px',
                          'borderLeft': '4px solid #9b59b6'})
            ])
        ], style={'marginBottom': '30px'}),

        html.Hr(),

        # QUICK SUMMARY CARDS
        html.Div([
            html.Div([
                html.H4("🕵️‍♂️ Purpose", style={'margin': '0', 'color': '#2c3e50'}),
                html.P("Root Cause Investigation", style={'margin': '5px 0', 'fontSize': '14px', 'color': '#7f8c8d'})
            ], style={'width': '24%', 'display': 'inline-block', 'padding': '15px', 'textAlign': 'center',
                      'backgroundColor': '#f4ecf7', 'margin': '5px', 'borderRadius': '8px'}),

            html.Div([
                html.H4("📊 Key Tests", style={'margin': '0', 'color': '#2c3e50'}),
                html.P("T-Test, ANOVA, Correlation", style={'margin': '5px 0', 'fontSize': '14px', 'color': '#7f8c8d'})
            ], style={'width': '24%', 'display': 'inline-block', 'padding': '15px', 'textAlign': 'center',
                      'backgroundColor': '#f4ecf7', 'margin': '5px', 'borderRadius': '8px'}),

            html.Div([
                html.H4("👥 Best For", style={'margin': '0', 'color': '#2c3e50'}),
                html.P("Process Engineers", style={'margin': '5px 0', 'fontSize': '14px', 'color': '#7f8c8d'})
            ], style={'width': '24%', 'display': 'inline-block', 'padding': '15px', 'textAlign': 'center',
                      'backgroundColor': '#f4ecf7', 'margin': '5px', 'borderRadius': '8px'}),

            html.Div([
                html.H4("🔬 Frequency", style={'margin': '0', 'color': '#2c3e50'}),
                html.P("Issue Investigation", style={'margin': '5px 0', 'fontSize': '14px', 'color': '#7f8c8d'})
            ], style={'width': '24%', 'display': 'inline-block', 'padding': '15px', 'textAlign': 'center',
                      'backgroundColor': '#f4ecf7', 'margin': '5px', 'borderRadius': '8px'}),
        ], style={'marginBottom': '30px'}),

        # Test results overview
        html.Div([
            html.Div([
                html.H4("Defective vs Non-Defective", style={'textAlign': 'center'}),
                dcc.Graph(
                    figure=create_hypothesis_gauge(test_results['defect_t_test']['p_value'], alpha,
                                                   "Defect Comparison"),
                    style={'height': '250px'}
                )
            ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px'}),

            html.Div([
                html.H4("Tool Differences", style={'textAlign': 'center'}),
                dcc.Graph(
                    figure=create_hypothesis_gauge(test_results['tool_anova']['p_value'], alpha,
                                                   "Tool ANOVA"),
                    style={'height': '250px'}
                )
            ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px'}),

            html.Div([
                html.H4("Normality", style={'textAlign': 'center'}),
                dcc.Graph(
                    figure=create_normality_gauge(test_results['normality']['p_value']),
                    style={'height': '250px'}
                )
            ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px'}),

            html.Div([
                html.H4("Correlation with Defects", style={'textAlign': 'center'}),
                dcc.Graph(
                    figure=create_correlation_gauge(test_results['defect_correlation']['correlation']),
                    style={'height': '250px'}
                )
            ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px'}),
        ]),

        # Detailed test results
        html.Div([
            html.H3("Detailed Test Results"),
            html.Div([
                html.Div([
                    html.H4("Defective vs Non-Defective T-Test"),
                    html.P(f"t-statistic: {test_results['defect_t_test']['t_stat']:.4f}"),
                    html.P(f"p-value: {test_results['defect_t_test']['p_value']:.4f}"),
                    html.P(f"Defective Mean: {test_results['defect_t_test']['defective_mean']:.4f}"),
                    html.P(f"Non-Defective Mean: {test_results['defect_t_test']['non_defective_mean']:.4f}"),
                    html.P(f"Conclusion: {test_results['defect_t_test']['conclusion']}"),
                ], style={'width': '32%', 'display': 'inline-block', 'padding': '10px',
                          'backgroundColor': '#f8f9fa', 'margin': '5px', 'borderRadius': '5px'}),

                html.Div([
                    html.H4("Tool ANOVA Test"),
                    html.P(f"F-statistic: {test_results['tool_anova']['f_stat']:.4f}"),
                    html.P(f"p-value: {test_results['tool_anova']['p_value']:.4f}"),
                    html.P(f"Conclusion: {test_results['tool_anova']['conclusion']}"),
                ], style={'width': '32%', 'display': 'inline-block', 'padding': '10px',
                          'backgroundColor': '#f8f9fa', 'margin': '5px', 'borderRadius': '5px'}),

                html.Div([
                    html.H4("Normality Test (Shapiro-Wilk)"),
                    html.P(f"W-statistic: {test_results['normality']['w_stat']:.4f}"),
                    html.P(f"p-value: {test_results['normality']['p_value']:.4f}"),
                    html.P(f"Conclusion: {test_results['normality']['conclusion']}"),
                ], style={'width': '32%', 'display': 'inline-block', 'padding': '10px',
                          'backgroundColor': '#f8f9fa', 'margin': '5px', 'borderRadius': '5px'}),
            ]),
        ], style={'marginTop': 20}),

        # Visualization of test results
        html.Div([
            dcc.Graph(
                figure=create_hypothesis_visualizations(global_df, parameter)
            )
        ], style={'marginTop': 20}),

        # Correlation matrix
        html.Div([
            html.H3("Parameter Correlation Matrix"),
            dcc.Graph(
                figure=create_correlation_heatmap(global_df)
            )
        ], style={'marginTop': 20})
    ])


def render_doe_tab():
    """Render Design of Experiments tab"""

    doe_results = perform_doe_analysis()

    return html.Div([
        html.H2("Design of Experiments Analysis"),
        html.Hr(),
        # STORYTELLING SECTION
        html.Div([
            html.Details([
                html.Summary("🚀 Launch Process Optimization (Click for Story & Summary)",
                             style={'cursor': 'pointer', 'color': '#2c3e50', 'fontWeight': 'bold', 'fontSize': '18px'}),
                html.Div([
                    html.Hr(),
                    html.P(["📖 " , html.Strong("The Story:")], style={'fontWeight': 'bold', 'color': '#34495e'}),
                    html.P([
                        "Suppose you've identified the key factors affecting your process through hypothesis testing.",
                        "Now, how do you optimize it? Welcome to your innovation lab. Instead of the old 'change one ",
                        "factor at a time' approach (slow and often misleading), we use structured Design of Experiments.",
                        "Imagine you're a chef perfecting a recipe: you systematically vary ingredients ",
                        "(Downforce, Platen Speed, Slurry Flow) to find the perfect combination for taste, texture,",
                        " and appearance (Roughness, Removal Rate, Uniformity). The main effects plots show you which",
                        " 'ingredient' has the biggest impact, while interaction plots reveal surprising synergies. ",
                        " The result? Data-backed recipe settings that take your process from 'good' to 'exceptional'."]),

                    html.Hr(),
                    html.P(["🔬 " , html.Strong("The Optimization Engine:")],
                           style={'fontWeight': 'bold', 'color': '#34495e'}),
                    html.Ol([
                        html.Li([ html.Strong("Main Effects Analysis: "),
                                "Which knob turns the volume up the most? Identify the most influential parameters."] ),
                        html.Li( [html.Strong("Interaction Mapping: "),
                                "Do factors work together or against each other? Find synergistic combinations."]),
                        html.Li([html.Strong("Multi-Objective Optimization: "),
                                "Balance competing goals—perfect roughness vs maximum throughput."]),
                        html.Li([ html.Strong("Optimal Recipe Generation: "),
                                "Get precise settings for your specific quality and productivity targets."])
                    ]),

                    html.Hr(),
                    html.P(["📋 " , html.Strong("Summary:")], style={'fontWeight': 'bold', 'color': '#34495e'}),
                    html.Ul([
                        html.Li([ html.Strong("Purpose: "),
                                "Systematically find the optimal process settings by understanding factor effects and interactions."]),
                        html.Li([html.Strong("Key Features: "),
                                "Main effects analysis, interaction plots, optimal settings recommendations, design matrix"]),
                        html.Li([html.Strong("Best For: "),
                                "Process engineers, integration engineers, and R&D focused on process optimization and improvement."]),
                        html.Li([html.Strong("Key Question: "),
                                "'What are the ideal process settings to achieve my quality and productivity targets?'"])
                    ]),

                    html.Hr(),
                    html.P(["💡 " , html.Strong("Pro Tip:")], style={'fontWeight': 'bold', 'color': '#34495e'}),
                    html.P([
                        "The further a gauge is from zero, the stronger that factor's effect. Red zones indicate ",
                        "powerful levers for process improvement. Use the 'Balanced Optimal' settings when you need to",
                        " optimize multiple responses simultaneously."],
                        style={'fontStyle': 'italic', 'backgroundColor': '#fff3cd', 'padding': '10px',
                               'borderRadius': '5px'})
                ], style={'padding': '20px', 'backgroundColor': '#e8f4f8', 'borderRadius': '10px',
                          'borderLeft': '4px solid #e67e22'})
            ])
        ], style={'marginBottom': '30px'}),

        html.Hr(),

        # QUICK SUMMARY CARDS
        html.Div([
            html.Div([
                html.H4("🚀 Purpose", style={'margin': '0', 'color': '#2c3e50'}),
                html.P("Process Optimization", style={'margin': '5px 0', 'fontSize': '14px', 'color': '#7f8c8d'})
            ], style={'width': '24%', 'display': 'inline-block', 'padding': '15px', 'textAlign': 'center',
                      'backgroundColor': '#fbeee6', 'margin': '5px', 'borderRadius': '8px'}),

            html.Div([
                html.H4("🎛️ Key Factors", style={'margin': '0', 'color': '#2c3e50'}),
                html.P("Downforce, Speed, Flow", style={'margin': '5px 0', 'fontSize': '14px', 'color': '#7f8c8d'})
            ], style={'width': '24%', 'display': 'inline-block', 'padding': '15px', 'textAlign': 'center',
                      'backgroundColor': '#fbeee6', 'margin': '5px', 'borderRadius': '8px'}),

            html.Div([
                html.H4("👥 Best For", style={'margin': '0', 'color': '#2c3e50'}),
                html.P("R&D & Optimization", style={'margin': '5px 0', 'fontSize': '14px', 'color': '#7f8c8d'})
            ], style={'width': '24%', 'display': 'inline-block', 'padding': '15px', 'textAlign': 'center',
                      'backgroundColor': '#fbeee6', 'margin': '5px', 'borderRadius': '8px'}),

            html.Div([
                html.H4("📈 Frequency", style={'margin': '0', 'color': '#2c3e50'}),
                html.P("Process Improvement", style={'margin': '5px 0', 'fontSize': '14px', 'color': '#7f8c8d'})
            ], style={'width': '24%', 'display': 'inline-block', 'padding': '15px', 'textAlign': 'center',
                      'backgroundColor': '#fbeee6', 'margin': '5px', 'borderRadius': '8px'}),
        ], style={'marginBottom': '30px'}),
        # DOE Overview
        html.Div([
            html.Div([
                html.H4("Optimal Roughness", style={'textAlign': 'center'}),
                html.H3(f"{doe_results['optimal_roughness']:.3f} Å",
                        style={'textAlign': 'center', 'color': '#27ae60'}),
                html.P("Target: < 1.3 Å", style={'textAlign': 'center'})
            ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px',
                      'backgroundColor': '#f8f9fa', 'margin': '5px', 'borderRadius': '5px'}),

            html.Div([
                html.H4("Optimal Removal Rate", style={'textAlign': 'center'}),
                html.H3(f"{doe_results['optimal_removal_rate']:.0f} nm/min",
                        style={'textAlign': 'center', 'color': '#3498db'}),
                html.P("Target: > 150 nm/min", style={'textAlign': 'center'})
            ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px',
                      'backgroundColor': '#f8f9fa', 'margin': '5px', 'borderRadius': '5px'}),

            html.Div([
                html.H4("Optimal Uniformity", style={'textAlign': 'center'}),
                html.H3(f"{doe_results['optimal_uniformity']:.3f}",
                        style={'textAlign': 'center', 'color': '#e74c3c'}),
                html.P("Target: > 0.95", style={'textAlign': 'center'})
            ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px',
                      'backgroundColor': '#f8f9fa', 'margin': '5px', 'borderRadius': '5px'}),

            html.Div([
                html.H4("Experiments", style={'textAlign': 'center'}),
                html.H3(f"{len(doe_results['design_matrix'])}",
                        style={'textAlign': 'center', 'color': '#f39c12'}),
                html.P("Full Factorial + Center", style={'textAlign': 'center'})
            ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px',
                      'backgroundColor': '#f8f9fa', 'margin': '5px', 'borderRadius': '5px'}),
        ]),

        # Main effects gauges
        html.Div([
            html.H3("Main Effects Analysis"),
            html.Div([
                html.Div([
                    dcc.Graph(
                        figure=create_effect_gauge(doe_results['main_effects']['A'], "Downforce Effect"),
                        style={'height': '250px'}
                    )
                ], style={'width': '32%', 'display': 'inline-block', 'padding': '10px'}),

                html.Div([
                    dcc.Graph(
                        figure=create_effect_gauge(doe_results['main_effects']['B'], "Platen Speed Effect"),
                        style={'height': '250px'}
                    )
                ], style={'width': '32%', 'display': 'inline-block', 'padding': '10px'}),

                html.Div([
                    dcc.Graph(
                        figure=create_effect_gauge(doe_results['main_effects']['C'], "Slurry Flow Effect"),
                        style={'height': '250px'}
                    )
                ], style={'width': '32%', 'display': 'inline-block', 'padding': '10px'}),
            ]),
        ], style={'marginTop': 20}),

        # DOE Visualizations
        html.Div([
            dcc.Graph(
                figure=create_doe_plots(doe_results)
            )
        ], style={'marginTop': 20}),

        # Optimal settings
        html.Div([
            html.H3("Recommended Optimal Settings"),
            html.Div([
                html.Div([
                    html.H4("For Minimum Roughness"),
                    html.P(f"Downforce: {doe_results['optimal_settings']['roughness']['downforce']}"),
                    html.P(f"Platen Speed: {doe_results['optimal_settings']['roughness']['platen_speed']}"),
                    html.P(f"Slurry Flow: {doe_results['optimal_settings']['roughness']['slurry_flow']}"),
                ], style={'width': '32%', 'display': 'inline-block', 'padding': '10px',
                          'backgroundColor': '#e8f5e8', 'margin': '5px', 'borderRadius': '5px'}),

                html.Div([
                    html.H4("For Maximum Removal Rate"),
                    html.P(f"Downforce: {doe_results['optimal_settings']['removal_rate']['downforce']}"),
                    html.P(f"Platen Speed: {doe_results['optimal_settings']['removal_rate']['platen_speed']}"),
                    html.P(f"Slurry Flow: {doe_results['optimal_settings']['removal_rate']['slurry_flow']}"),
                ], style={'width': '32%', 'display': 'inline-block', 'padding': '10px',
                          'backgroundColor': '#e8f5e8', 'margin': '5px', 'borderRadius': '5px'}),

                html.Div([
                    html.H4("Balanced Optimal"),
                    html.P(f"Downforce: {doe_results['optimal_settings']['balanced']['downforce']}"),
                    html.P(f"Platen Speed: {doe_results['optimal_settings']['balanced']['platen_speed']}"),
                    html.P(f"Slurry Flow: {doe_results['optimal_settings']['balanced']['slurry_flow']}"),
                ], style={'width': '32%', 'display': 'inline-block', 'padding': '10px',
                          'backgroundColor': '#e8f5e8', 'margin': '5px', 'borderRadius': '5px'}),
            ]),
        ], style={'marginTop': 20}),

        # Design matrix table
        html.Div([
            html.H3("Design Matrix"),
            dash_table.DataTable(
                data=doe_results['design_matrix'].to_dict('records'),
                columns=[{"name": i, "id": i} for i in doe_results['design_matrix'].columns],
                page_size=10,
                style_cell={'textAlign': 'center'},
                style_header={'backgroundColor': '#2c3e50', 'color': 'white', 'fontWeight': 'bold'}
            )
        ], style={'marginTop': 20})
    ])


def render_overview_tab():
    """Render process overview tab"""

    return html.Div([
        html.H2("📊 Process Overview & Real-time Monitoring"),

        # STORYTELLING SECTION
        html.Div([
            html.Details([
                html.Summary("🎯 Process Overview Details (Click for Story & Summary)",
                             style={'cursor': 'pointer', 'color': '#2c3e50', 'fontWeight': 'bold', 'fontSize': '18px'}),
                html.Div([
                    html.Hr(),
                    html.P(html.Strong("📖 The Story:"), style={'fontWeight': 'bold', 'color': '#34495e'}),
                    html.P([
                        "This is the mission control. This simulates a real-time pulse of the production line.",
                        "You can immediately spot if any tool is behaving erratically through the control charts,",
                        "compare performance across tools to identify outliers, and monitor the most critical quality",
                        "metric - the defect rate—over time. It's designed for a quick, comprehensive health check to ",
                        "answer the question: Is my process running smoothly and consistently right now?"
                    ]),

                    html.Hr(),
                    html.P(html.Strong("📋 Summary:"), style={'fontWeight': 'bold', 'color': '#34495e'}),
                    html.Ul([
                        html.Li([html.Strong("Purpose: "),
                                "Real-time monitoring and high-level process health assessment."]),
                        html.Li( [ html.Strong("Key Features: "),
                                "Control Charts, Tool Performance Comparison, Defect Trend Analysis"] ),
                        html.Li([ html.Strong("Best For: "),
                                "Shift engineers, line managers, and anyone needing an at-a-glance status update."]),
                        html.Li([ html.Strong("Key Question: "), "'Is everything running normally right now?'"])
                    ])
                ], style={'padding': '20px', 'backgroundColor': '#e8f4f8', 'borderRadius': '10px',
                          'borderLeft': '4px solid #3498db'})
            ])
        ], style={'marginBottom': '30px'}),

        html.Hr(),

        # QUICK SUMMARY CARDS
        html.Div([
            html.Div([
                html.H4("🎯 Purpose", style={'margin': '0', 'color': '#2c3e50'}),
                html.P("Real-time Health Check", style={'margin': '5px 0', 'fontSize': '14px', 'color': '#7f8c8d'})
            ], style={'width': '24%', 'display': 'inline-block', 'padding': '15px', 'textAlign': 'center',
                      'backgroundColor': '#e8f4f8', 'margin': '5px', 'borderRadius': '8px'}),

            html.Div([
                html.H4("📊 Key Metrics", style={'margin': '0', 'color': '#2c3e50'}),
                html.P("Stability, Tool Performance, Defect Rates",
                       style={'margin': '5px 0', 'fontSize': '14px', 'color': '#7f8c8d'})
            ], style={'width': '24%', 'display': 'inline-block', 'padding': '15px', 'textAlign': 'center',
                      'backgroundColor': '#e8f4f8', 'margin': '5px', 'borderRadius': '8px'}),

            html.Div([
                html.H4("👥 Best For", style={'margin': '0', 'color': '#2c3e50'}),
                html.P("Shift Engineers & Managers", style={'margin': '5px 0', 'fontSize': '14px', 'color': '#7f8c8d'})
            ], style={'width': '24%', 'display': 'inline-block', 'padding': '15px', 'textAlign': 'center',
                      'backgroundColor': '#e8f4f8', 'margin': '5px', 'borderRadius': '8px'}),

            html.Div([
                html.H4("⏱️ Frequency", style={'margin': '0', 'color': '#2c3e50'}),
                html.P("Daily/Shift Review", style={'margin': '5px 0', 'fontSize': '14px', 'color': '#7f8c8d'})
            ], style={'width': '24%', 'display': 'inline-block', 'padding': '15px', 'textAlign': 'center',
                      'backgroundColor': '#e8f4f8', 'margin': '5px', 'borderRadius': '8px'}),
        ], style={'marginBottom': '30px'}),

        # Control charts
        html.Div([
            dcc.Graph(
                figure=create_control_charts(global_df)
            )
        ]),

        # Tool performance
        html.Div([
            html.H3("Tool Performance Comparison"),
            html.Div([
                html.Div([
                    dcc.Graph(
                        figure=create_tool_performance_chart(global_df, 'thickness')
                    )
                ], style={'width': '49%', 'display': 'inline-block'}),

                html.Div([
                    dcc.Graph(
                        figure=create_tool_performance_chart(global_df, 'roughness')
                    )
                ], style={'width': '49%', 'display': 'inline-block'}),
            ]),
        ], style={'marginTop': 20}),

        # Defect analysis
        html.Div([
            html.H3("Defect Analysis"),
            html.Div([
                html.Div([
                    dcc.Graph(
                        figure=create_defect_trend_chart(global_df)
                    )
                ], style={'width': '49%', 'display': 'inline-block'}),

                html.Div([
                    dcc.Graph(
                        figure=create_defect_by_tool_chart(global_df)
                    )
                ], style={'width': '49%', 'display': 'inline-block'}),
            ]),
        ], style={'marginTop': 20}),

        # Recent data
        html.Div([
            html.H3("Recent Process Data"),
            dash_table.DataTable(
                data=global_df.tail(10).to_dict('records'),
                columns=[{"name": i, "id": i} for i in global_df.columns],
                style_cell={'textAlign': 'center'},
                style_header={'backgroundColor': '#2c3e50', 'color': 'white', 'fontWeight': 'bold'}
            )
        ], style={'marginTop': 20})
    ])


@app.callback(
    Output('tab-info-panel', 'children'),
    [Input('main-tabs', 'value')]
)
def update_tab_info(selected_tab):
    tab_stories = {
        'overview-tab': [
            html.Strong("📊 Your Mission Control"),
            html.Br(),
            html.Small("Real-time process monitoring and quick health checks."),
            html.Br(),
            html.Small(" Answer: 'Is my process running smoothly right now?'"),
        ],
        'capability-tab': [
            html.Strong("🎯 Process Capability"),
            html.Br(),
            html.Small("Quantify specification compliance."),
            html.Br(),
            html.Small("Answer: 'Is my process capable of meeting customer requirements?'",),
        ],
        'hypothesis-tab': [
            html.Strong("🔍 Root Cause Detective"),
            html.Br(),
            html.Small("Statistical evidence for investigations."),
            html.Br(),
            html.Small("Answer: 'Why are we seeing these defects or differences?'"),
        ],
        'doe-tab': [
            html.Strong("⚗️ Process Optimization"),
            html.Br(),
            html.Small("Find optimal recipe settings. "),
            html.Small("Answer: 'How can I improve my process performance?'"),
        ]
    }
    return tab_stories.get(selected_tab, [html.Small("Select a tab for guidance", style=SIDEBAR_LABEL_STYLE)])

@app.callback(
    Output("welcome-modal", "is_open"),
    [Input("close-welcome", "n_clicks")],
)
def close_modal(n_clicks):
    if n_clicks:
        return False
    return True


# ADD THIS CALLBACK to handle the refresh functionality
@app.callback(
    [Output('tab-content', 'children', allow_duplicate=True),
     Output('parameter-selector', 'value')],  # Reset parameter selector if needed
    [Input('refresh-button', 'n_clicks')],
    [dash.dependencies.State('main-tabs', 'value'),
     dash.dependencies.State('parameter-selector', 'value'),
     dash.dependencies.State('confidence-level', 'value')],
    prevent_initial_call=True
)
def refresh_data(n_clicks, selected_tab, selected_param, confidence_level):
    """Refresh data and reload the current tab"""
    if n_clicks > 0:
        # Generate new data with a random seed based on current time
        import time
        new_seed = int(time.time()) % 1000000
        global global_df
        global_df = generate_cmp_data(seed=new_seed)

        # Re-render the current tab with new data
        if selected_tab == 'capability-tab':
            return render_capability_tab(selected_param), selected_param
        elif selected_tab == 'hypothesis-tab':
            return render_hypothesis_tab(selected_param, confidence_level), selected_param
        elif selected_tab == 'doe-tab':
            return render_doe_tab(), selected_param
        elif selected_tab == 'overview-tab':
            return render_overview_tab(), selected_param

    # If no refresh, return current state
    return dash.no_update, dash.no_update


@app.callback(
    [Output("defect-rate-metric", "children"),
     Output("wafers-processed-metric", "children"),
     Output("tools-active-metric", "children"),
     Output("lots-completed-metric", "children")],
    [Input('refresh-button', 'n_clicks')]
)
def update_metrics(n_clicks):
    defect_rate = f"{global_df['defect_flag'].mean() * 100:.2f}%"
    wafers_processed = f"{len(global_df)}"
    tools_active = f"{global_df['tool_id'].nunique()}"
    lots_completed = f"{global_df['lot_id'].nunique()}"

    return defect_rate, wafers_processed, tools_active, lots_completed


# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .metric-card {
                background: white;
                padding: 5px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                border-left: 4px solid #3498db;
            }
            body {
                background-color: #f8f9fa;
                font-family: Arial, sans-serif;
            }
            .tab-content {
                background: white;
                padding: 5px;
                border-radius: 2px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    # Get port from environment variable or default to 8050
    port = int(os.environ.get("PORT", 8050))

    # Run with production settings
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False  # Set to False in production
    )