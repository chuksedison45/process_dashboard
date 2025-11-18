# 🔬 CMP Process Analytics Dashboard
## 📋 Overview
The CMP Process Analytics Dashboard is a simulated web application built with Dash to 
showcase real-time monitoring, statistical analysis, and optimization capabilities of a 
Chemical Mechanical Planarization (CMP) processes in semiconductor manufacturing. 
This dashboard transforms raw process data into actionable intelligence through four 
interconnected analytical modules.

![Status](https://img.shields.io/badge/Status-Production_Ready-green) 
![Python](https://img.shields.io/badge/Python-3.13%2B-blue) 
![Framework](https://img.shields.io/badge/Framework-Dash-ff69b4)

## 🎯 Key Features
### 📊 Process Overview
- Real-time Monitoring: Control charts for key parameters (Thickness, Roughness, Downforce, Platen RPM)
- Tool Performance: Comparative analysis across manufacturing tools
- Defect Analytics: Trend analysis and tool-specific defect rates
- Production Metrics: Live counters for wafers processed, active tools, and completed lots

### 🎯 Process Capability
- Cpk/Cp Analysis: Gauge visualizations for process capability indices
- Yield Calculations: Real-time yield percentage and PPM (Parts Per Million) metrics
- Specification Compliance: Distribution analysis against USL/LSL limits
- Statistical Summary: Comprehensive parameter capability reporting

### 🔍 Hypothesis Testing
- Statistical Validation: T-tests, ANOVA, and correlation analysis
- Defect Root Cause: Identify significant parameter differences between defective/non-defective wafers
- Tool Performance: Statistical significance testing across manufacturing tools
- Normality Testing: Shapiro-Wilk tests for data distribution validation

### ⚗️ Design of Experiments (DOE)
- Factor Optimization: Main effects analysis for Downforce, Platen Speed, and Slurry Flow
- Response Optimization: Balanced settings for Roughness, Removal Rate, and Uniformity
- Interaction Analysis: Heatmap visualization of parameter interactions
- Optimal Recipes: Data-driven recommendations for process improvement

## 🚀 Installation & Setup
### Prerequisites
- Python 3.13 or higher
- pip (Python package manager)
- dash==2.14.2
- plotly==5.17.0
- pandas==2.1.4
- numpy==1.24.4
- scipy==1.11.4
- gunicorn==21.2.0
- dash-bootstrap-components==1.5.0


### Installation Steps
- Clone or download the project

```bash
git clone https://github.com/chuksedison45/process_dashboard.git
cd process_dashboard
```
- Create a virtual environment (recommended)
```bash
python -m venv cmp_env
source cmp_env/bin/activate  # On Windows: cmp_env\Scripts\activate
```
- Install required packages (dependencies)
```bash
pip install -r requirements.txt
```
- Run the dashboard
```bash
python cmp_process_dashboard.py
```

- Access the dashboard

  - Open your web browser 
  - Navigate to: http://127.0.0.1:8050

📁 Project Structure
text
process_dashboard/
├── cmp_process_dashboard.py  # Main application file
├── requirements.txt                         # Dependencies list
├── README.md                               # This file
└── assets/                                 # Static files (CSS, images)


## 🔧 Technical Architecture
### Frontend
- Dash: Core web framework
- Plotly: Interactive visualizations and charts
- Dash Bootstrap Components: UI layout and styling
- HTML/CSS: Custom styling and responsive design

### Backend
- Pandas: Data manipulation and analysis
- NumPy: Numerical computations
- SciPy: Statistical testing and analysis
- Scikit-learn: Machine learning utilities (if extended)

### Data Flow
- Data Generation: Synthetic CMP process data with realistic variations
- Processing: Statistical calculations and capability analysis
- Visualization: Interactive charts and gauges
- User Interaction: Real-time updates based on parameter selection

## 🎨 Dashboard Navigation
### Sidebar Controls
- Parameter Selector: Choose process parameters for analysis
- Confidence Level: Set statistical significance level (90%, 95%, 99%)
- Refresh Button: Regenerate data and update analyses
- Tab Guide: Contextual information for each analytical module

### Main Tabs
- 📊 Process Overview: Real-time monitoring and health checks
- 🎯 Process Capability: Specification compliance and yield analysis
- 🔍 Hypothesis Testing: Statistical validation and root cause analysis
- ⚗️ Design of Experiments: Process optimization and parameter tuning

### 📈 Key Metrics & Specifications
- Process Parameters

| Parameter  | Target	 |  USL	  | LSL	 | Unit  |
|:----------:|:-------:|:------:|:----:|:-----:|
| Thickness	 |  725.0  | 725.5	 | 724.5	  | nm |
| Roughness  |  1.2   |  	1.8	  |   0.5   |      	Å      |
|Downforce|	4.5|	5.2|	3.8	|psi|
|Platen RPM	|120|	135|	105|	rpm|
|Slurry Flow|	250|	290	|210	|ml/min|

- Quality Standards 
  - Target Defect Rate: < 1.5% 
  - Cpk Acceptance: ≥ 1.33 
  - Yield Target: > 99% 
  - PPM Target: < 10,000

### 🔍 Use Cases
- For Process Engineers 
  - Monitor real-time process stability
  - Identify tool-specific performance issues 
  - Validate process changes and improvements 
  - Optimize recipe parameters for yield enhancement

- For Quality Engineers 
  - Track process capability indices 
  - Perform statistical process control 
  - Conduct root cause analysis for defects 
  - Generate quality reports and compliance documentation

- For Manufacturing Managers 
  - Overview production line performance 
  - Monitor overall equipment effectiveness (OEE)
  - Make data-driven decisions for capacity planning 
  - Track key performance indicators (KPIs)

## 🛠️ Customization Guide
- Adding New Parameters 
  - Update generate_cmp_data() function 
  - Add specifications to specs dictionary 
  - Include in parameter selector dropdown 
  - Create relevant visualization functions

- Modifying Statistical Methods 
  - Update perform_hypothesis_tests() for different statistical tests 
  - Modify perform_doe_analysis() for custom experimental designs 
  - Adjust confidence levels and significance thresholds

- Styling Customization 
  - Modify SIDEBAR_STYLE and CONTENT_STYLE dictionaries
  - Update color schemes in gauge and chart functions 
  - Customize CSS in the app.index_string section

## 📊 Sample Data Structure
The dashboard uses synthetic data with the following structure:

```python
{
    'timestamp': datetime_index,
    'wafer_id': unique_identifier,
    'lot_id': batch_grouping,
    'thickness': normal_distribution,
    'roughness': normal_distribution,
    'downforce_psi': normal_distribution,
    'platen_rpm': normal_distribution,
    'slurry_flow': normal_distribution,
    'pad_age_hours': random_integers,
    'defect_flag': binary_classification,
    'tool_id': categorical_labels
}
```

## 🚀 Performance Optimization
- For Large Datasets 
  - Implement data caching with @cache.memoize 
  - Use database connections instead of in-memory DataFrames 
  - Add pagination for data tables 
  - Implement background callbacks for heavy computations

- Deployment Considerations 
  - Use production WSGI server (Gunicorn)
  - Implement proper logging and error handling 
  - Add user authentication and authorization 
  - Set up monitoring and alerting

## 🤝 Contributing
We welcome contributions to enhance the dashboard:

- Fork the repository
- Create a feature branch (git checkout -b feature/improvement)
- Commit your changes (`git commit -am 'Add new feature'`)
- Push to the branch (git push origin feature/improvement)
- Create a Pull Request

## 📞 Support & Documentation
### 🛠️ Troubleshooting
- Port already in use: Change port in app.run(port=8051)
- Missing dependencies: Run pip install -r requirements.txt
- Chart rendering issues: Check browser console for JavaScript errors

### Additional Resources
- [Dash Documentation](https://dash.plotly.com/)
- [Plotly Python Graphing Library](https://plotly.com/python/)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
- Intel Corporation for the CMP process domain knowledge
- Plotly team for the excellent Dash framework
- Semiconductor manufacturing community for best practices and insights

