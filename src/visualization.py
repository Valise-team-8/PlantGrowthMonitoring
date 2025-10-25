import pandas as pd
import matplotlib.pyplot as plt

def plot_growth(csv_path="models/results.csv"):
    df = pd.read_csv(csv_path)
    
    # Group by day and calculate statistics
    daily_stats = df.groupby('day').agg({
        'area': ['mean', 'std', 'count'],
        'height': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    daily_stats.columns = ['day', 'area_mean', 'area_std', 'area_count', 'height_mean', 'height_std', 'height_count']
    
    # Extract day numbers for plotting
    daily_stats['day_num'] = daily_stats['day'].str.extract('(\d+)').astype(int)
    daily_stats = daily_stats.sort_values('day_num')
    
    # Create three subplots: individual metrics + combined progress
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: Area with error bars
    ax1 = plt.subplot(2, 2, 1)
    ax1.errorbar(daily_stats["day_num"], daily_stats["area_mean"], 
                yerr=daily_stats["area_std"], label="Plant Area", 
                marker='o', linewidth=3, capsize=5, color='#2E8B57', markersize=8)
    ax1.set_title("Plant Area Growth Over Time", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Day", fontsize=12)
    ax1.set_ylabel("Area (pixels)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(daily_stats["day_num"])
    
    # Plot 2: Height with error bars
    ax2 = plt.subplot(2, 2, 2)
    ax2.errorbar(daily_stats["day_num"], daily_stats["height_mean"], 
                yerr=daily_stats["height_std"], label="Plant Height", 
                marker='s', linewidth=3, capsize=5, color='#FF8C00', markersize=8)
    ax2.set_title("Plant Height Growth Over Time", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Day", fontsize=12)
    ax2.set_ylabel("Height (pixels)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(daily_stats["day_num"])
    
    # Plot 3: Combined Progress Line Plot (normalized)
    ax3 = plt.subplot(2, 1, 2)
    
    # Normalize values to show relative growth (0-100%)
    area_normalized = ((daily_stats["area_mean"] - daily_stats["area_mean"].min()) / 
                      (daily_stats["area_mean"].max() - daily_stats["area_mean"].min()) * 100)
    height_normalized = ((daily_stats["height_mean"] - daily_stats["height_mean"].min()) / 
                        (daily_stats["height_mean"].max() - daily_stats["height_mean"].min()) * 100)
    
    ax3.plot(daily_stats["day_num"], area_normalized, 
             marker='o', linewidth=4, markersize=10, color='#2E8B57', 
             label='Plant Area Progress', alpha=0.8)
    ax3.plot(daily_stats["day_num"], height_normalized, 
             marker='s', linewidth=4, markersize=10, color='#FF8C00', 
             label='Plant Height Progress', alpha=0.8)
    
    # Fill area under curves for better visualization
    ax3.fill_between(daily_stats["day_num"], area_normalized, alpha=0.2, color='#2E8B57')
    ax3.fill_between(daily_stats["day_num"], height_normalized, alpha=0.2, color='#FF8C00')
    
    ax3.set_title("🌱 Plant Growth Progress Over Time", fontsize=16, fontweight='bold')
    ax3.set_xlabel("Day", fontsize=14)
    ax3.set_ylabel("Growth Progress (%)", fontsize=14)
    ax3.set_ylim(-5, 105)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=12, loc='upper left')
    ax3.set_xticks(daily_stats["day_num"])
    
    # Add progress annotations
    for i, row in daily_stats.iterrows():
        ax3.annotate(f'{area_normalized.iloc[i]:.0f}%', 
                    (row['day_num'], area_normalized.iloc[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center', 
                    fontsize=10, color='#2E8B57', fontweight='bold')
        ax3.annotate(f'{height_normalized.iloc[i]:.0f}%', 
                    (row['day_num'], height_normalized.iloc[i]), 
                    textcoords="offset points", xytext=(0,-15), ha='center', 
                    fontsize=10, color='#FF8C00', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate growth rates
    print(f"\n📊 Growth Analysis:")
    print(f"{'Day':<6} {'Area (avg)':<12} {'Height (avg)':<14} {'Area Change':<12} {'Height Change':<14}")
    print("-" * 70)
    
    prev_area = None
    prev_height = None
    
    for _, row in daily_stats.iterrows():
        area_change = ""
        height_change = ""
        
        if prev_area is not None:
            area_diff = row['area_mean'] - prev_area
            height_diff = row['height_mean'] - prev_height
            area_change = f"{area_diff:+.0f} ({area_diff/prev_area*100:+.1f}%)"
            height_change = f"{height_diff:+.0f} ({height_diff/prev_height*100:+.1f}%)"
        
        print(f"{row['day']:<6} {row['area_mean']:<12.0f} {row['height_mean']:<14.0f} {area_change:<12} {height_change:<14}")
        
        prev_area = row['area_mean']
        prev_height = row['height_mean']
    
    # Summary
    total_area_growth = ((daily_stats["area_mean"].iloc[-1] - daily_stats["area_mean"].iloc[0]) / 
                        daily_stats["area_mean"].iloc[0] * 100)
    total_height_growth = ((daily_stats["height_mean"].iloc[-1] - daily_stats["height_mean"].iloc[0]) / 
                          daily_stats["height_mean"].iloc[0] * 100)
    
    print(f"\n🌿 Overall Growth Summary:")
    print(f"   Total Area Growth: {total_area_growth:+.1f}%")
    print(f"   Total Height Growth: {total_height_growth:+.1f}%")

def plot_growth_web(csv_path="models/results.csv", output_path="static/growth_plot.png"):
    """
    Generate growth plots for web display (saves to file instead of showing).
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    df = pd.read_csv(csv_path)
    
    # Group by day and calculate statistics
    daily_stats = df.groupby('day').agg({
        'area': ['mean', 'std', 'count'],
        'height': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    daily_stats.columns = ['day', 'area_mean', 'area_std', 'area_count', 'height_mean', 'height_std', 'height_count']
    
    # Extract day numbers for plotting
    daily_stats['day_num'] = daily_stats['day'].str.extract('(\d+)').astype(int)
    daily_stats = daily_stats.sort_values('day_num')
    
    # Create figure for web
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Area with error bars
    ax1 = plt.subplot(2, 2, 1)
    ax1.errorbar(daily_stats["day_num"], daily_stats["area_mean"], 
                yerr=daily_stats["area_std"], label="Plant Area", 
                marker='o', linewidth=3, capsize=5, color='#2E8B57', markersize=8)
    ax1.set_title("Plant Area Growth Over Time", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Day", fontsize=12)
    ax1.set_ylabel("Area (pixels)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(daily_stats["day_num"])
    
    # Plot 2: Height with error bars
    ax2 = plt.subplot(2, 2, 2)
    ax2.errorbar(daily_stats["day_num"], daily_stats["height_mean"], 
                yerr=daily_stats["height_std"], label="Plant Height", 
                marker='s', linewidth=3, capsize=5, color='#FF8C00', markersize=8)
    ax2.set_title("Plant Height Growth Over Time", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Day", fontsize=12)
    ax2.set_ylabel("Height (pixels)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(daily_stats["day_num"])
    
    # Plot 3: Combined Progress Line Plot (normalized)
    ax3 = plt.subplot(2, 1, 2)
    
    # Normalize values to show relative growth (0-100%)
    area_normalized = ((daily_stats["area_mean"] - daily_stats["area_mean"].min()) / 
                      (daily_stats["area_mean"].max() - daily_stats["area_mean"].min()) * 100)
    height_normalized = ((daily_stats["height_mean"] - daily_stats["height_mean"].min()) / 
                        (daily_stats["height_mean"].max() - daily_stats["height_mean"].min()) * 100)
    
    ax3.plot(daily_stats["day_num"], area_normalized, 
             marker='o', linewidth=4, markersize=10, color='#2E8B57', 
             label='Plant Area Progress', alpha=0.8)
    ax3.plot(daily_stats["day_num"], height_normalized, 
             marker='s', linewidth=4, markersize=10, color='#FF8C00', 
             label='Plant Height Progress', alpha=0.8)
    
    # Fill area under curves for better visualization
    ax3.fill_between(daily_stats["day_num"], area_normalized, alpha=0.2, color='#2E8B57')
    ax3.fill_between(daily_stats["day_num"], height_normalized, alpha=0.2, color='#FF8C00')
    
    ax3.set_title("🌱 Plant Growth Progress Over Time", fontsize=16, fontweight='bold')
    ax3.set_xlabel("Day", fontsize=14)
    ax3.set_ylabel("Growth Progress (%)", fontsize=14)
    ax3.set_ylim(-5, 105)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=12, loc='upper left')
    ax3.set_xticks(daily_stats["day_num"])
    
    # Add progress annotations
    for i, row in daily_stats.iterrows():
        ax3.annotate(f'{area_normalized.iloc[i]:.0f}%', 
                    (row['day_num'], area_normalized.iloc[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center', 
                    fontsize=10, color='#2E8B57', fontweight='bold')
        ax3.annotate(f'{height_normalized.iloc[i]:.0f}%', 
                    (row['day_num'], height_normalized.iloc[i]), 
                    textcoords="offset points", xytext=(0,-15), ha='center', 
                    fontsize=10, color='#FF8C00', fontweight='bold')
    
    plt.tight_layout()
    
    # Save to file instead of showing
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()  # Close the figure to free memory
    
    print(f"📊 Web plot saved to {output_path}")