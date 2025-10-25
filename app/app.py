from flask import Flask, render_template_string, send_file, jsonify
import matplotlib.pyplot as plt
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import from src after path setup
try:
    import src.analysis as analysis
except ImportError:
    print("Error: Cannot import src.analysis. Make sure you're running from the project root directory.")
    sys.exit(1)

app = Flask(__name__)

# Cache for storing analysis results
CACHE = {
    'data': None,
    'plots_generated': False,
    'last_update': 0,
    'cache_duration': 300  # 5 minutes cache
}


def load_or_generate_data():
    """Load data from cache or generate if needed"""
    current_time = time.time()

    # Check if we have cached data and it's still valid
    if (CACHE['data'] is not None and
            current_time - CACHE['last_update'] < CACHE['cache_duration']):
        print("📊 Using cached data")
        return CACHE['data']

    print("🔄 Generating fresh analysis...")

    # Check if results.csv exists and is recent
    results_file = os.path.join(project_root, "models", "results.csv")

    if os.path.exists(results_file):
        # Use existing CSV if it exists
        df = pd.read_csv(results_file)
        print(f"📁 Loaded existing data: {len(df)} records")
    else:
        # Run analysis only if no CSV exists
        dataset_dir = os.path.join(project_root, "dataset")
        df = analysis.analyze_growth(dataset_dir, results_file)
        print(f"🔬 Generated new analysis: {len(df)} records")

    # Cache the data
    CACHE['data'] = df
    CACHE['last_update'] = current_time

    return df


def generate_plots_if_needed(df, static_dir):
    """Generate plots only if they don't exist or cache is invalid"""

    plot_files = ['area_plot.png', 'height_plot.png', 'progress_plot.png']
    plots_exist = all(os.path.exists(os.path.join(static_dir, f))
                      for f in plot_files)

    if plots_exist and CACHE['plots_generated']:
        print("📈 Using existing plots")
        return

    print("🎨 Generating plots...")

    # Group by day and calculate statistics
    daily_stats = df.groupby('day').agg({
        'area': ['mean', 'std', 'count'],
        'height': ['mean', 'std', 'count']
    }).reset_index()

    # Flatten column names
    daily_stats.columns = ['day', 'area_mean', 'area_std',
                           'area_count', 'height_mean', 'height_std', 'height_count']
    daily_stats['day_num'] = daily_stats['day'].str.extract(
        r'(\d+)').astype(int)
    daily_stats = daily_stats.sort_values('day_num')

    # Set dark theme for matplotlib
    plt.style.use('dark_background')

    # Generate compact plots with dark theme
    # 1. Compact Area Growth Plot
    fig, ax = plt.subplots(figsize=(5, 3), facecolor='#2d3748')
    ax.set_facecolor('#1a202c')
    ax.errorbar(daily_stats["day_num"], daily_stats["area_mean"],
                yerr=daily_stats["area_std"], marker='o', linewidth=2,
                capsize=4, color='#68d391', markersize=6, alpha=0.9)
    ax.fill_between(daily_stats["day_num"],
                    daily_stats["area_mean"] - daily_stats["area_std"],
                    daily_stats["area_mean"] + daily_stats["area_std"],
                    alpha=0.3, color='#68d391')
    ax.set_title("🌱 Area Growth", fontsize=11,
                 fontweight='bold', color='#e2e8f0')
    ax.set_xlabel("Days", fontsize=9, color='#a0aec0')
    ax.set_ylabel("Area", fontsize=9, color='#a0aec0')
    ax.grid(True, alpha=0.2, linestyle='--', color='#4a5568')
    ax.tick_params(colors='#a0aec0', labelsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'area_plot.png'), dpi=80, bbox_inches='tight',
                facecolor='#2d3748', edgecolor='none')
    plt.close()

    # 2. Compact Height Growth Plot
    fig, ax = plt.subplots(figsize=(5, 3), facecolor='#2d3748')
    ax.set_facecolor('#1a202c')
    ax.errorbar(daily_stats["day_num"], daily_stats["height_mean"],
                yerr=daily_stats["height_std"], marker='s', linewidth=2,
                capsize=4, color='#f6ad55', markersize=6, alpha=0.9)
    ax.fill_between(daily_stats["day_num"],
                    daily_stats["height_mean"] - daily_stats["height_std"],
                    daily_stats["height_mean"] + daily_stats["height_std"],
                    alpha=0.3, color='#f6ad55')
    ax.set_title("📏 Height Growth", fontsize=11,
                 fontweight='bold', color='#e2e8f0')
    ax.set_xlabel("Days", fontsize=9, color='#a0aec0')
    ax.set_ylabel("Height", fontsize=9, color='#a0aec0')
    ax.grid(True, alpha=0.2, linestyle='--', color='#4a5568')
    ax.tick_params(colors='#a0aec0', labelsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'height_plot.png'), dpi=80, bbox_inches='tight',
                facecolor='#2d3748', edgecolor='none')
    plt.close()

    # 3. Compact Progress Chart
    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='#2d3748')
    ax.set_facecolor('#1a202c')

    area_normalized = ((daily_stats["area_mean"] - daily_stats["area_mean"].min()) /
                       (daily_stats["area_mean"].max() - daily_stats["area_mean"].min()) * 100)
    height_normalized = ((daily_stats["height_mean"] - daily_stats["height_mean"].min()) /
                         (daily_stats["height_mean"].max() - daily_stats["height_mean"].min()) * 100)

    ax.plot(daily_stats["day_num"], area_normalized, marker='o', linewidth=2.5,
            markersize=8, color='#68d391', label='Area', alpha=0.9)
    ax.plot(daily_stats["day_num"], height_normalized, marker='s', linewidth=2.5,
            markersize=8, color='#f6ad55', label='Height', alpha=0.9)

    ax.fill_between(daily_stats["day_num"],
                    area_normalized, alpha=0.2, color='#68d391')
    ax.fill_between(daily_stats["day_num"],
                    height_normalized, alpha=0.2, color='#f6ad55')

    ax.set_title("🚀 Growth Progress", fontsize=11,
                 fontweight='bold', color='#e2e8f0')
    ax.set_xlabel("Days", fontsize=9, color='#a0aec0')
    ax.set_ylabel("Progress (%)", fontsize=9, color='#a0aec0')
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.2, linestyle='--', color='#4a5568')
    ax.legend(fontsize=9, facecolor='#2d3748',
              edgecolor='#4a5568', labelcolor='#e2e8f0')
    ax.tick_params(colors='#a0aec0', labelsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'progress_plot.png'), dpi=80, bbox_inches='tight',
                facecolor='#2d3748', edgecolor='none')
    plt.close()

    # Reset matplotlib style
    plt.rcdefaults()

    CACHE['plots_generated'] = True
    print("✅ Plots generated successfully")


@app.route('/')
def home():
    try:
        start_time = time.time()

        # Load data (cached or fresh)
        df = load_or_generate_data()

        # Create static directory
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        os.makedirs(static_dir, exist_ok=True)

        # Generate plots only if needed
        generate_plots_if_needed(df, static_dir)

        # Calculate daily stats (fast operation)
        daily_stats = df.groupby('day').agg({
            'area': ['mean', 'std', 'count'],
            'height': ['mean', 'std', 'count']
        }).reset_index()

        daily_stats.columns = ['day', 'area_mean', 'area_std',
                               'area_count', 'height_mean', 'height_std', 'height_count']
        daily_stats['day_num'] = daily_stats['day'].str.extract(
            r'(\d+)').astype(int)
        daily_stats = daily_stats.sort_values('day_num')

        # Calculate comprehensive growth metrics
        total_images = len(df)
        days_analyzed = len(daily_stats)

        # Growth rates
        area_growth = ((daily_stats["area_mean"].iloc[-1] - daily_stats["area_mean"].iloc[0]) /
                       daily_stats["area_mean"].iloc[0] * 100)
        height_growth = ((daily_stats["height_mean"].iloc[-1] - daily_stats["height_mean"].iloc[0]) /
                         daily_stats["height_mean"].iloc[0] * 100)

        # Advanced analytics
        avg_daily_area_change = area_growth / \
            (days_analyzed - 1) if days_analyzed > 1 else 0
        avg_daily_height_change = height_growth / \
            (days_analyzed - 1) if days_analyzed > 1 else 0

        # Variability metrics
        area_cv = (daily_stats["area_std"].mean() /
                   daily_stats["area_mean"].mean()) * 100
        height_cv = (daily_stats["height_std"].mean() /
                     daily_stats["height_mean"].mean()) * 100

        # Growth consistency (lower CV = more consistent)
        growth_consistency = max(0, 100 - (area_cv + height_cv) / 2)

        # Growth acceleration (change in growth rate)
        if len(daily_stats) >= 3:
            early_growth = daily_stats["area_mean"].iloc[1] - \
                daily_stats["area_mean"].iloc[0]
            late_growth = daily_stats["area_mean"].iloc[-1] - \
                daily_stats["area_mean"].iloc[-2]
            growth_acceleration = (
                (late_growth - early_growth) / early_growth * 100) if early_growth > 0 else 0
        else:
            growth_acceleration = 0

        # Health score (composite metric)
        health_score = max(
            0, min(100, 50 + (area_growth + height_growth) / 4 - (area_cv + height_cv) / 4))

        # Growth trend analysis
        area_trend = "Increasing" if area_growth > 5 else "Stable" if area_growth > - \
            5 else "Decreasing"
        height_trend = "Increasing" if height_growth > 5 else "Stable" if height_growth > - \
            5 else "Decreasing"

        # Statistical insights
        max_area_day = daily_stats.loc[daily_stats["area_mean"].idxmax(
        ), "day"]
        max_height_day = daily_stats.loc[daily_stats["height_mean"].idxmax(
        ), "day"]

        # Growth rate classification
        if area_growth > 20:
            growth_rate_class = "Excellent"
        elif area_growth > 10:
            growth_rate_class = "Good"
        elif area_growth > 0:
            growth_rate_class = "Moderate"
        else:
            growth_rate_class = "Poor"

        # Prepare data for charts
        chart_data = {
            'days': daily_stats['day_num'].tolist(),
            'area_means': daily_stats['area_mean'].tolist(),
            'height_means': daily_stats['height_mean'].tolist(),
            'area_stds': daily_stats['area_std'].tolist(),
            'height_stds': daily_stats['height_std'].tolist()
        }

        load_time = time.time() - start_time
        print(f"⚡ Page loaded in {load_time:.2f} seconds")

        return render_template_string(MODERN_TEMPLATE,
                                      daily_stats=daily_stats,
                                      total_images=total_images,
                                      days_analyzed=days_analyzed,
                                      area_growth=area_growth,
                                      height_growth=height_growth,
                                      avg_daily_area_change=avg_daily_area_change,
                                      avg_daily_height_change=avg_daily_height_change,
                                      area_cv=area_cv,
                                      height_cv=height_cv,
                                      health_score=health_score,
                                      growth_consistency=growth_consistency,
                                      growth_acceleration=growth_acceleration,
                                      area_trend=area_trend,
                                      height_trend=height_trend,
                                      max_area_day=max_area_day,
                                      max_height_day=max_height_day,
                                      growth_rate_class=growth_rate_class,
                                      chart_data=json.dumps(chart_data),
                                      load_time=load_time)

    except Exception as e:
        return render_template_string(ERROR_TEMPLATE, error=str(e))


@app.route('/api/data')
def api_data():
    """API endpoint for real-time data updates"""
    try:
        results_file = os.path.join(project_root, "models", "results.csv")
        df = pd.read_csv(results_file)

        daily_stats = df.groupby('day').agg({
            'area': ['mean', 'std', 'count'],
            'height': ['mean', 'std', 'count']
        }).reset_index()

        daily_stats.columns = ['day', 'area_mean', 'area_std',
                               'area_count', 'height_mean', 'height_std', 'height_count']
        daily_stats['day_num'] = daily_stats['day'].str.extract(
            r'(\d+)').astype(int)

        return jsonify({
            'success': True,
            'data': daily_stats.to_dict('records')
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/images/<day>')
def get_day_images(day):
    """Get sample images for a specific day"""
    try:
        dataset_dir = os.path.join(project_root, "dataset", day)
        if not os.path.exists(dataset_dir):
            return jsonify({'success': False, 'error': f'Day {day} not found'})

        # Get all images for the day
        images = [f for f in os.listdir(
            dataset_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Sample a few images from each category for display
        sample_images = []
        categories = ['Healthy', 'EarlyBloom',
                      'MatureBud', 'YoungBud', 'Wilted']

        for category in categories:
            category_images = [
                img for img in images if img.startswith(category)]
            if category_images:
                # Take first 2 images from each category
                sample_images.extend(category_images[:2])

        # If no categorized images, just take first 10 images
        if not sample_images:
            sample_images = images[:10]

        return jsonify({
            'success': True,
            'day': day,
            'images': sample_images[:10],  # Limit to 10 images for performance
            'total_images': len(images)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/dataset/<day>/<filename>')
def serve_image(day, filename):
    """Serve individual plant images"""
    try:
        dataset_dir = os.path.join(project_root, "dataset", day)
        return send_file(os.path.join(dataset_dir, filename))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/static/<filename>')
def static_files(filename):
    return send_file(os.path.join('static', filename))


# Modern HTML Template with Agricultural Theme
MODERN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌾 AgriGrowth Analytics - Plant Monitoring Dashboard</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">

    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a2332 0%, #2d3748 100%);
            min-height: 100vh;
            color: #e2e8f0;
        }
        
        .header {
            background: linear-gradient(135deg, #2f855a 0%, #38a169 100%);
            color: white;
            padding: 1.5rem 0;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        
        .header h1 {
            font-size: 3rem;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 1rem;
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border-left: 4px solid;
            color: #e2e8f0;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.15);
        }
        
        .metric-card.green { border-left-color: #68d391; }
        .metric-card.orange { border-left-color: #f6ad55; }
        .metric-card.blue { border-left-color: #63b3ed; }
        .metric-card.purple { border-left-color: #b794f6; }
        
        .metric-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            opacity: 0.8;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        
        .metric-label {
            color: #666;
            font-size: 1.1rem;
        }
        
        .chart-section {
            background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }
        
        .chart-title {
            font-size: 1.5rem;
            margin-bottom: 1rem;
            color: #68d391;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .plot-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 0.8rem;
            margin-bottom: 1rem;
        }
        
        .analysis-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-bottom: 1rem;
        }
        
        .insight-card {
            background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            border-left: 3px solid;
        }
        
        .insight-card.trend { border-left-color: #9f7aea; }
        .insight-card.stats { border-left-color: #4fd1c7; }
        .insight-card.performance { border-left-color: #f093fb; }
        
        .plot-container {
            background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            text-align: center;
        }
        
        .plot-container img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .data-table {
            background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            overflow-x: auto;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }
        
        th, td {
            padding: 1rem;
            text-align: center;
            border-bottom: 1px solid #eee;
        }
        
        th {
            background: linear-gradient(135deg, #2f855a 0%, #38a169 100%);
            color: white;
            font-weight: 600;
        }
        
        tr:hover {
            background-color: rgba(104, 211, 145, 0.1);
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background-color: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 0.5rem;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
            transition: width 0.3s ease;
        }
        
        .interactive-chart {
            height: 300px;
            margin-top: 1rem;
        }
        
        .explanation-box {
            background: rgba(45, 55, 72, 0.6);
            border-left: 4px solid #68d391;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 0 8px 8px 0;
            backdrop-filter: blur(10px);
        }
        
        .explanation-box h4 {
            color: #68d391;
            margin-bottom: 0.5rem;
            font-size: 1rem;
        }
        
        .explanation-box p {
            font-size: 0.9rem;
            line-height: 1.4;
        }
        
        .footer {
            text-align: center;
            padding: 2rem;
            color: white;
            background: rgba(0,0,0,0.1);
            margin-top: 3rem;
            border-radius: 15px;
        }
        
        @media (max-width: 768px) {
            .header h1 { font-size: 2rem; }
            .dashboard-grid { grid-template-columns: 1fr; }
            .plot-grid { grid-template-columns: 1fr; }
            .container { padding: 1rem; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1><i class="fas fa-seedling"></i> AgriGrowth Analytics</h1>
        <p>Advanced Plant Growth Monitoring & Analysis Dashboard</p>
    </div>
    
    <div class="container">
        <!-- Metrics Dashboard -->
        <div class="dashboard-grid">
            <div class="metric-card green">
                <div class="metric-icon" style="color: #68d391;">
                    <i class="fas fa-seedling"></i>
                </div>
                <div class="metric-value">{{ "%.1f"|format(area_growth) }}%</div>
                <div class="metric-label">Total Area Growth</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {{ (area_growth + 100) / 2 }}%;"></div>
                </div>
            </div>
            
            <div class="metric-card orange">
                <div class="metric-icon" style="color: #f6ad55;">
                    <i class="fas fa-arrows-alt-v"></i>
                </div>
                <div class="metric-value">{{ "%.1f"|format(height_growth) }}%</div>
                <div class="metric-label">Total Height Growth</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {{ (height_growth + 100) / 2 }}%;"></div>
                </div>
            </div>
            
            <div class="metric-card blue">
                <div class="metric-icon" style="color: #63b3ed;">
                    <i class="fas fa-heartbeat"></i>
                </div>
                <div class="metric-value">{{ "%.0f"|format(health_score) }}</div>
                <div class="metric-label">Plant Health Score</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {{ health_score }}%;"></div>
                </div>
            </div>
            
            <div class="metric-card purple">
                <div class="metric-icon" style="color: #b794f6;">
                    <i class="fas fa-chart-line"></i>
                </div>
                <div class="metric-value">{{ "%.0f"|format(growth_consistency) }}</div>
                <div class="metric-label">Growth Consistency</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {{ growth_consistency }}%;"></div>
                </div>
            </div>
            
            <div class="metric-card green">
                <div class="metric-icon" style="color: #68d391;">
                    <i class="fas fa-rocket"></i>
                </div>
                <div class="metric-value">{{ "%.1f"|format(growth_acceleration) }}%</div>
                <div class="metric-label">Growth Acceleration</div>
            </div>
            
            <div class="metric-card orange">
                <div class="metric-icon" style="color: #f6ad55;">
                    <i class="fas fa-award"></i>
                </div>
                <div class="metric-value">{{ growth_rate_class }}</div>
                <div class="metric-label">Growth Rating</div>
            </div>
        </div>
        
        <!-- Static Plots Only -->
        
        <!-- Static Plots Grid -->
        <div class="plot-grid">
            <div class="plot-container">
                <h3 style="color: #2E8B57; margin-bottom: 1rem;">
                    <i class="fas fa-leaf"></i> Plant Area Analysis
                </h3>
                <div class="explanation-box">
                    <h4>Area Growth Insights</h4>
                    <p>Plant area measurement helps track leaf development and photosynthetic capacity. Larger area typically indicates healthier plant growth and better light absorption.</p>
                </div>
                <img src="/static/area_plot.png" alt="Plant Area Growth">
            </div>
            
            <div class="plot-container">
                <h3 style="color: #FF8C00; margin-bottom: 1rem;">
                    <i class="fas fa-ruler-vertical"></i> Plant Height Analysis
                </h3>
                <div class="explanation-box">
                    <h4>Height Growth Insights</h4>
                    <p>Height measurement tracks vertical development and structural growth. Consistent height increase indicates proper stem development and overall plant vigor.</p>
                </div>
                <img src="/static/height_plot.png" alt="Plant Height Growth">
            </div>
        </div>
        
        <div class="plot-container" style="margin-bottom: 1rem;">
            <h3 style="color: #6f42c1; margin-bottom: 1rem;">
                <i class="fas fa-rocket"></i> Growth Progress Tracker
            </h3>
            <div class="explanation-box">
                <h4>Progress Analysis</h4>
                <p>Normalized view showing relative growth progress. Helps identify growth patterns and acceleration/deceleration trends.</p>
            </div>
            <img src="/static/progress_plot.png" alt="Growth Progress">
        </div>
        
        <!-- Plant Image Gallery -->
        <div class="chart-section">
            <h2 class="chart-title">
                <i class="fas fa-images"></i>
                Plant Image Gallery
            </h2>
            <div class="explanation-box">
                <h4><i class="fas fa-camera"></i> Visual Growth Documentation</h4>
                <p>Browse actual plant images from each day to see visual growth progression. Images are categorized by plant health status: Healthy, Early Bloom, Mature Bud, Young Bud, and Wilted.</p>
            </div>
            
            <div class="day-selector" style="margin-bottom: 1rem;">
                <label for="daySelect" style="color: #e2e8f0; margin-right: 1rem;">Select Day:</label>
                <select id="daySelect" style="padding: 0.5rem; border-radius: 5px; border: none; background: #4a5568; color: #e2e8f0;">
                    {% for _, row in daily_stats.iterrows() %}
                    <option value="{{ row['day'] }}">{{ row['day'] }} ({{ row['area_count'] }} images)</option>
                    {% endfor %}
                </select>
                <button id="loadImages" style="
                    background: linear-gradient(135deg, #2f855a 0%, #38a169 100%);
                    color: white; border: none; padding: 0.5rem 1rem; border-radius: 5px;
                    cursor: pointer; margin-left: 0.5rem;
                ">Load Images</button>
            </div>
            
            <div id="imageGallery" style="
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin-top: 1rem;
            ">
                <div style="text-align: center; color: #a0aec0; grid-column: 1 / -1;">
                    <i class="fas fa-images" style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.5;"></i>
                    <p>Select a day above to view plant images</p>
                </div>
            </div>
        </div>
        
        <!-- Advanced Analytics Section -->
        <div class="analysis-grid">
            <div class="insight-card trend">
                <h3 style="color: #9f7aea; margin-bottom: 1rem;">
                    <i class="fas fa-trending-up"></i> Growth Trends
                </h3>
                <div style="margin-bottom: 0.8rem;">
                    <strong>Area Trend:</strong> 
                    <span style="color: {% if area_trend == 'Increasing' %}#68d391{% elif area_trend == 'Stable' %}#f6ad55{% else %}#fc8181{% endif %};">
                        {{ area_trend }}
                    </span>
                </div>
                <div style="margin-bottom: 0.8rem;">
                    <strong>Height Trend:</strong> 
                    <span style="color: {% if height_trend == 'Increasing' %}#68d391{% elif height_trend == 'Stable' %}#f6ad55{% else %}#fc8181{% endif %};">
                        {{ height_trend }}
                    </span>
                </div>
                <div style="margin-bottom: 0.8rem;">
                    <strong>Peak Area Day:</strong> {{ max_area_day }}
                </div>
                <div>
                    <strong>Peak Height Day:</strong> {{ max_height_day }}
                </div>
            </div>
            
            <div class="insight-card stats">
                <h3 style="color: #4fd1c7; margin-bottom: 1rem;">
                    <i class="fas fa-calculator"></i> Statistical Insights
                </h3>
                <div style="margin-bottom: 0.8rem;">
                    <strong>Area Variability:</strong> {{ "%.1f"|format(area_cv) }}%
                    <div class="progress-bar" style="margin-top: 0.3rem;">
                        <div class="progress-fill" style="width: {{ area_cv }}%; background: #4fd1c7;"></div>
                    </div>
                </div>
                <div style="margin-bottom: 0.8rem;">
                    <strong>Height Variability:</strong> {{ "%.1f"|format(height_cv) }}%
                    <div class="progress-bar" style="margin-top: 0.3rem;">
                        <div class="progress-fill" style="width: {{ height_cv }}%; background: #4fd1c7;"></div>
                    </div>
                </div>
                <div style="margin-bottom: 0.8rem;">
                    <strong>Daily Area Change:</strong> {{ "%.2f"|format(avg_daily_area_change) }}%
                </div>
                <div>
                    <strong>Daily Height Change:</strong> {{ "%.2f"|format(avg_daily_height_change) }}%
                </div>
            </div>
            
            <div class="insight-card performance">
                <h3 style="color: #f093fb; margin-bottom: 1rem;">
                    <i class="fas fa-chart-pie"></i> Performance Metrics
                </h3>
                <div style="margin-bottom: 0.8rem;">
                    <strong>Images Processed:</strong> {{ total_images }}
                </div>
                <div style="margin-bottom: 0.8rem;">
                    <strong>Days Monitored:</strong> {{ days_analyzed }}
                </div>
                <div style="margin-bottom: 0.8rem;">
                    <strong>Avg Images/Day:</strong> {{ "%.0f"|format(total_images / days_analyzed) }}
                </div>
                <div>
                    <strong>Data Quality:</strong> 
                    <span style="color: {% if growth_consistency > 80 %}#68d391{% elif growth_consistency > 60 %}#f6ad55{% else %}#fc8181{% endif %};">
                        {% if growth_consistency > 80 %}Excellent{% elif growth_consistency > 60 %}Good{% else %}Fair{% endif %}
                    </span>
                </div>
            </div>
        </div>
        
        <!-- Growth Interpretation Guide -->
        <div class="chart-section">
            <h2 class="chart-title">
                <i class="fas fa-lightbulb"></i>
                Growth Analysis Guide
            </h2>
            <div class="analysis-grid">
                <div class="explanation-box">
                    <h4><i class="fas fa-seedling"></i> Area Growth Interpretation</h4>
                    <p><strong>High Growth (>20%):</strong> Excellent leaf development, strong photosynthetic capacity</p>
                    <p><strong>Moderate Growth (5-20%):</strong> Normal healthy development</p>
                    <p><strong>Low Growth (<5%):</strong> May indicate stress or maturity phase</p>
                </div>
                <div class="explanation-box">
                    <h4><i class="fas fa-arrows-alt-v"></i> Height Growth Interpretation</h4>
                    <p><strong>Consistent Height Growth:</strong> Good structural development</p>
                    <p><strong>Variable Height:</strong> May indicate environmental stress</p>
                    <p><strong>Plateau Phase:</strong> Normal for mature plants</p>
                </div>
                <div class="explanation-box">
                    <h4><i class="fas fa-chart-line"></i> Variability Analysis</h4>
                    <p><strong>Low Variability (<10%):</strong> Consistent growing conditions</p>
                    <p><strong>High Variability (>20%):</strong> Environmental stress or measurement noise</p>
                    <p><strong>Growth Acceleration:</strong> Indicates improving conditions</p>
                </div>
            </div>
        </div>
        
        <!-- Data Table -->
        <div class="data-table">
            <h2 class="chart-title">
                <i class="fas fa-table"></i>
                Detailed Growth Statistics
            </h2>
            <div class="explanation-box">
                <h4>Statistical Summary</h4>
                <p>This table provides detailed daily statistics including mean values, standard deviations, and sample counts. The data helps identify trends, variability, and statistical significance of growth measurements.</p>
            </div>
            <table>
                <thead>
                    <tr>
                        <th><i class="fas fa-calendar"></i> Day</th>
                        <th><i class="fas fa-expand-arrows-alt"></i> Avg Area (pixels)</th>
                        <th><i class="fas fa-arrows-alt-v"></i> Avg Height (pixels)</th>
                        <th><i class="fas fa-images"></i> Sample Count</th>
                        <th><i class="fas fa-chart-bar"></i> Area Std Dev</th>
                        <th><i class="fas fa-chart-bar"></i> Height Std Dev</th>
                    </tr>
                </thead>
                <tbody>
                    {% for _, row in daily_stats.iterrows() %}
                    <tr>
                        <td><strong>{{ row['day'] }}</strong></td>
                        <td>{{ "%.0f"|format(row['area_mean']) }}</td>
                        <td>{{ "%.0f"|format(row['height_mean']) }}</td>
                        <td>{{ row['area_count'] }}</td>
                        <td>{{ "%.1f"|format(row['area_std']) }}</td>
                        <td>{{ "%.1f"|format(row['height_std']) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p><i class="fas fa-leaf"></i> Powered by AgriGrowth Analytics | Advanced Plant Monitoring Technology</p>
            <p style="margin-top: 0.5rem; opacity: 0.8;">
                <i class="fas fa-clock"></i> Last updated: <span id="lastUpdate"></span> | 
                <i class="fas fa-tachometer-alt"></i> Load time: {{ "%.2f"|format(load_time) }}s
            </p>
            <button id="refreshBtn" onclick="refreshData()" style="
                background: linear-gradient(135deg, #2f855a 0%, #38a169 100%);
                color: white; border: none; padding: 0.5rem 1rem; border-radius: 8px;
                cursor: pointer; margin-top: 1rem; font-size: 0.9rem;
                transition: all 0.3s ease;
            " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                <i class="fas fa-sync-alt"></i> Refresh Data
            </button>
        </div>
    </div>
    
    <script>
        // Static plots only - no interactive chart
        
        // Update timestamp
        document.getElementById('lastUpdate').textContent = new Date().toLocaleString();
        
        // Simple refresh without page reload
        function refreshData() {
            document.getElementById('refreshBtn').innerHTML = '<i class="fas fa-spinner fa-spin"></i> Refreshing...';
            document.getElementById('lastUpdate').textContent = new Date().toLocaleString();
            setTimeout(() => {
                document.getElementById('refreshBtn').innerHTML = '<i class="fas fa-sync-alt"></i> Data Refreshed';
                setTimeout(() => {
                    document.getElementById('refreshBtn').innerHTML = '<i class="fas fa-sync-alt"></i> Refresh Data';
                }, 2000);
            }, 500);
        }
        
        // Image gallery functionality
        document.getElementById('loadImages').addEventListener('click', function() {
            const selectedDay = document.getElementById('daySelect').value;
            const gallery = document.getElementById('imageGallery');
            
            // Show loading
            gallery.innerHTML = '<div style="text-align: center; color: #a0aec0; grid-column: 1 / -1;"><i class="fas fa-spinner fa-spin" style="font-size: 2rem;"></i><p>Loading images...</p></div>';
            
            fetch(`/images/${selectedDay}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        displayImages(data.day, data.images, data.total_images);
                    } else {
                        gallery.innerHTML = '<div style="text-align: center; color: #fc8181; grid-column: 1 / -1;"><i class="fas fa-exclamation-triangle" style="font-size: 2rem;"></i><p>Error loading images: ' + data.error + '</p></div>';
                    }
                })
                .catch(error => {
                    gallery.innerHTML = '<div style="text-align: center; color: #fc8181; grid-column: 1 / -1;"><i class="fas fa-exclamation-triangle" style="font-size: 2rem;"></i><p>Error loading images</p></div>';
                });
        });
        
        function displayImages(day, images, totalImages) {
            const gallery = document.getElementById('imageGallery');
            
            let html = `<div style="grid-column: 1 / -1; text-align: center; margin-bottom: 1rem; color: #68d391;">
                <h3><i class="fas fa-calendar"></i> ${day} - Showing ${images.length} of ${totalImages} images</h3>
            </div>`;
            
            images.forEach(image => {
                const category = image.split('_')[0];
                const categoryColor = getCategoryColor(category);
                
                html += `
                    <div style="
                        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
                        border-radius: 10px;
                        padding: 1rem;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                        transition: transform 0.3s ease;
                        border-left: 4px solid ${categoryColor};
                    " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                        <img src="/dataset/${day}/${image}" 
                             alt="${image}" 
                             style="
                                width: 100%; 
                                height: 150px; 
                                object-fit: cover; 
                                border-radius: 8px;
                                margin-bottom: 0.5rem;
                             "
                             onclick="openImageModal('/dataset/${day}/${image}', '${image}')">
                        <div style="text-align: center;">
                            <div style="color: ${categoryColor}; font-weight: bold; font-size: 0.9rem;">
                                ${category}
                            </div>
                            <div style="color: #a0aec0; font-size: 0.8rem; margin-top: 0.2rem;">
                                ${image.replace(/\s*\(Small\)/, '')}
                            </div>
                        </div>
                    </div>
                `;
            });
            
            gallery.innerHTML = html;
        }
        
        function getCategoryColor(category) {
            const colors = {
                'Healthy': '#68d391',
                'EarlyBloom': '#f6ad55',
                'MatureBud': '#9f7aea',
                'YoungBud': '#63b3ed',
                'Wilted': '#fc8181'
            };
            return colors[category] || '#a0aec0';
        }
        
        function openImageModal(imageSrc, imageName) {
            // Create modal
            const modal = document.createElement('div');
            modal.style.cssText = `
                position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                background: rgba(0,0,0,0.9); z-index: 1000; display: flex;
                align-items: center; justify-content: center; cursor: pointer;
            `;
            
            modal.innerHTML = `
                <div style="max-width: 90%; max-height: 90%; text-align: center;">
                    <img src="${imageSrc}" style="max-width: 100%; max-height: 80vh; border-radius: 10px;">
                    <div style="color: white; margin-top: 1rem; font-size: 1.2rem;">${imageName}</div>
                    <div style="color: #a0aec0; margin-top: 0.5rem;">Click anywhere to close</div>
                </div>
            `;
            
            modal.onclick = () => document.body.removeChild(modal);
            document.body.appendChild(modal);
        }
    </script>
</body>
</html>
"""

ERROR_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AgriGrowth Analytics - Error</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f8f9fa; padding: 2rem; }
        .error-container { max-width: 600px; margin: 0 auto; background: white; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        .error-icon { font-size: 4rem; color: #dc3545; text-align: center; margin-bottom: 1rem; }
        h1 { color: #dc3545; text-align: center; }
        .error-message { background: #f8d7da; color: #721c24; padding: 1rem; border-radius: 5px; margin: 1rem 0; }
    </style>
</head>
<body>
    <div class="error-container">
        <div class="error-icon">⚠️</div>
        <h1>Oops! Something went wrong</h1>
        <div class="error-message">
            <strong>Error:</strong> {{ error }}
        </div>
        <p style="text-align: center;">
            <a href="/" style="color: #28a745; text-decoration: none;">← Back to Dashboard</a>
        </p>
    </div>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
