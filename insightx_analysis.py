
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings, os

warnings.filterwarnings("ignore")

# ── GLOBAL STYLE ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font="DejaVu Sans")
PALETTE   = ["#185FA5", "#1D9E75", "#D85A30", "#BA7517", "#7F77DD", "#D4537E"]
BG        = "#F8F9FB"
CARD_BG   = "white"
CHART_DIR = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(CHART_DIR, exist_ok=True)

def save(name):
    path = os.path.join(CHART_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ✓ Saved {name}")


# 1. LOAD & CLEAN

print("\n── 1. Loading & Cleaning ──")

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "ecommerce_data.csv"),
                 parse_dates=["order_date"])

# Introduce realistic nulls for demo cleaning
np.random.seed(0)
null_idx = np.random.choice(df.index, size=int(len(df)*0.02), replace=False)
df.loc[null_idx[:len(null_idx)//2], "gender"]    = np.nan
df.loc[null_idx[len(null_idx)//2:], "age_group"] = np.nan

print(f"  Raw shape     : {df.shape}")
print(f"  Null counts   :\n{df.isnull().sum()[df.isnull().sum()>0]}")

# Fill missing categoricals with mode
df["gender"].fillna(df["gender"].mode()[0], inplace=True)
df["age_group"].fillna(df["age_group"].mode()[0], inplace=True)

# Derived columns
df["month"]        = df["order_date"].dt.to_period("M").astype(str)
df["month_num"]    = df["order_date"].dt.month
df["quarter"]      = df["order_date"].dt.to_period("Q").astype(str)
df["day_of_week"]  = df["order_date"].dt.day_name()
df["profit_margin"] = ((df["revenue"] / (df["unit_price"] * df["quantity"])) - 1 + df["discount"]).round(4)

# Filter only completed orders for revenue analysis
df_completed = df[df["status"] == "Completed"].copy()

print(f"  Clean shape   : {df.shape}")
print(f"  Date range    : {df['order_date'].min().date()} → {df['order_date'].max().date()}")
print(f"  Total revenue : ₹{df_completed['revenue'].sum():,.0f}")
print(f"  Unique customers: {df['customer_id'].nunique()}")


# 2. SUMMARY STATISTICS

print("\n── 2. Summary Statistics ──")

print("\n  Revenue by Category:")
cat_stats = df_completed.groupby("category")["revenue"].agg(["sum","mean","count"]).round(2)
cat_stats.columns = ["Total Revenue","Avg Order Value","Orders"]
cat_stats = cat_stats.sort_values("Total Revenue", ascending=False)
print(cat_stats.to_string())

print("\n  Order Status Distribution:")
print(df["status"].value_counts(normalize=True).mul(100).round(1).to_string())


# 3. CHART 1 — Monthly Revenue Trend

print("\n── 3. Chart: Monthly Revenue Trend ──")

monthly = (df_completed.groupby("month")["revenue"]
           .sum().reset_index().rename(columns={"revenue":"total_revenue"}))
monthly["rolling_3m"] = monthly["total_revenue"].rolling(3, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
ax.set_facecolor(BG)
ax.fill_between(monthly["month"], monthly["total_revenue"],
                alpha=0.15, color=PALETTE[0])
ax.plot(monthly["month"], monthly["total_revenue"],
        marker="o", color=PALETTE[0], linewidth=2.2, markersize=6, label="Monthly Revenue")
ax.plot(monthly["month"], monthly["rolling_3m"],
        linestyle="--", color=PALETTE[2], linewidth=1.8, label="3-Month Rolling Avg")

ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
ax.set_xticks(range(len(monthly)))
ax.set_xticklabels(monthly["month"], rotation=45, ha="right", fontsize=9)
ax.set_title("Monthly Revenue Trend — 2023", fontsize=14, fontweight="bold", pad=14)
ax.set_xlabel("")
ax.legend(framealpha=0.8)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
save("01_monthly_revenue_trend.png")


# 4. CHART 2 — Revenue by Category (horizontal bar)

print("── 4. Chart: Revenue by Category ──")

cat_rev = (df_completed.groupby("category")["revenue"]
           .sum().sort_values().reset_index())

fig, ax = plt.subplots(figsize=(9, 4), facecolor=BG)
ax.set_facecolor(BG)
bars = ax.barh(cat_rev["category"], cat_rev["revenue"],
               color=PALETTE[:len(cat_rev)], edgecolor="white", height=0.6)
for bar, val in zip(bars, cat_rev["revenue"]):
    ax.text(bar.get_width() + 5000, bar.get_y() + bar.get_height()/2,
            f"₹{val/1000:.0f}K", va="center", fontsize=10, color="#444")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
ax.set_title("Total Revenue by Product Category", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Revenue (₹)")
ax.spines[["top","right","left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
plt.tight_layout()
save("02_revenue_by_category.png")


# 5. CHART 3 — Sales Channel Performance

print("── 5. Chart: Sales Channel Performance ──")

channel = (df_completed.groupby("channel")
           .agg(revenue=("revenue","sum"), orders=("order_id","count"))
           .reset_index().sort_values("revenue", ascending=False))
channel["aov"] = (channel["revenue"] / channel["orders"]).round(0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=BG)
for ax in axes: ax.set_facecolor(BG)

# Revenue bars
axes[0].bar(channel["channel"], channel["revenue"],
            color=PALETTE[:len(channel)], edgecolor="white")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
axes[0].set_title("Revenue by Channel", fontsize=12, fontweight="bold")
axes[0].set_xlabel("")
axes[0].tick_params(axis="x", rotation=20)
axes[0].spines[["top","right"]].set_visible(False)

# AOV bars
axes[1].bar(channel["channel"], channel["aov"],
            color=PALETTE[:len(channel)], edgecolor="white", alpha=0.8)
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:.0f}"))
axes[1].set_title("Avg Order Value by Channel", fontsize=12, fontweight="bold")
axes[1].tick_params(axis="x", rotation=20)
axes[1].spines[["top","right"]].set_visible(False)

plt.suptitle("Sales Channel Performance", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
save("03_channel_performance.png")


# 6. CHART 4 — Customer Segments (Age × Gender heatmap)

print("── 6. Chart: Customer Segment Heatmap ──")

seg = (df_completed.groupby(["age_group","gender"])["revenue"]
       .sum().unstack(fill_value=0))
age_order = ["18-24","25-34","35-44","45-54","55+"]
seg = seg.reindex(age_order)

fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=BG)
sns.heatmap(seg/1000, annot=True, fmt=".0f", cmap="Blues",
            linewidths=0.5, linecolor="#eee",
            cbar_kws={"label":"Revenue (₹K)"}, ax=ax)
ax.set_title("Revenue Heatmap — Age Group × Gender", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Gender")
ax.set_ylabel("Age Group")
plt.tight_layout()
save("04_customer_segment_heatmap.png")


# 7. CHART 5 — Top 10 Products by Revenue

print("── 7. Chart: Top 10 Products ──")

top_products = (df_completed.groupby(["category","product"])["revenue"]
                .sum().reset_index()
                .sort_values("revenue", ascending=False).head(10))

cat_color_map = dict(zip(df["category"].unique(), PALETTE))
colors = [cat_color_map[c] for c in top_products["category"]]

fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
ax.set_facecolor(BG)
bars = ax.barh(top_products["product"], top_products["revenue"],
               color=colors, edgecolor="white", height=0.65)
for bar, val in zip(bars, top_products["revenue"]):
    ax.text(bar.get_width() + 1500, bar.get_y() + bar.get_height()/2,
            f"₹{val/1000:.1f}K", va="center", fontsize=9.5, color="#444")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
ax.set_title("Top 10 Products by Revenue", fontsize=14, fontweight="bold", pad=12)
ax.spines[["top","right","left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
# Legend for categories
from matplotlib.patches import Patch
legend_items = [Patch(facecolor=cat_color_map[c], label=c) for c in top_products["category"].unique()]
ax.legend(handles=legend_items, title="Category", fontsize=9, title_fontsize=9,
          loc="lower right", framealpha=0.8)
plt.tight_layout()
save("05_top10_products.png")


# 8. CHART 6 — Order Status Donut

print("── 8. Chart: Order Status Donut ──")

status_counts = df["status"].value_counts()
explode = [0.04] * len(status_counts)
status_colors = [PALETTE[0], PALETTE[2], PALETTE[3]]

fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
ax.set_facecolor(BG)
wedges, texts, autotexts = ax.pie(
    status_counts, labels=status_counts.index,
    autopct="%1.1f%%", colors=status_colors,
    startangle=140, explode=explode,
    wedgeprops={"edgecolor":"white","linewidth":2},
    pctdistance=0.78, labeldistance=1.1
)
for t in autotexts: t.set_fontsize(10); t.set_color("white"); t.set_fontweight("bold")
centre = plt.Circle((0,0), 0.52, color=BG)
ax.add_artist(centre)
ax.text(0, 0, f"{len(df):,}\nOrders", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#333")
ax.set_title("Order Status Distribution", fontsize=13, fontweight="bold", pad=10)
plt.tight_layout()
save("06_order_status_donut.png")


# 9. CHART 7 — Revenue by Region × Quarter

print("── 9. Chart: Region × Quarter Revenue ──")

rq = (df_completed.groupby(["region","quarter"])["revenue"]
      .sum().reset_index())
pivot_rq = rq.pivot(index="region", columns="quarter", values="revenue").fillna(0)

fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
ax.set_facecolor(BG)
pivot_rq.plot(kind="bar", ax=ax, color=PALETTE[:4], edgecolor="white", width=0.72)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
ax.set_title("Quarterly Revenue by Region", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Region")
ax.set_ylabel("Revenue (₹)")
ax.tick_params(axis="x", rotation=0)
ax.spines[["top","right"]].set_visible(False)
ax.legend(title="Quarter", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
plt.tight_layout()
save("07_region_quarter_revenue.png")


# 10. CHART 8 — Price vs Revenue scatter (coloured by category)

print("── 10. Chart: Price vs Revenue Scatter ──")

fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
ax.set_facecolor(BG)
for idx, (cat, grp) in enumerate(df_completed.groupby("category")):
    ax.scatter(grp["unit_price"], grp["revenue"],
               alpha=0.35, s=22, color=PALETTE[idx % len(PALETTE)], label=cat)
ax.set_xlabel("Unit Price (₹)", fontsize=11)
ax.set_ylabel("Order Revenue (₹)", fontsize=11)
ax.set_title("Unit Price vs Order Revenue", fontsize=14, fontweight="bold", pad=12)
ax.legend(title="Category", fontsize=9, title_fontsize=9, markerscale=1.5,
          bbox_to_anchor=(1.01, 1), loc="upper left")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
save("08_price_vs_revenue_scatter.png")


# 11. CHART 9 — Day-of-Week Order Volume

print("── 11. Chart: Day-of-Week Order Volume ──")

dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dow = (df.groupby("day_of_week")["order_id"]
       .count().reindex(dow_order).reset_index())
dow.columns = ["day","orders"]

fig, ax = plt.subplots(figsize=(9, 4), facecolor=BG)
ax.set_facecolor(BG)
bar_colors = [PALETTE[2] if d in ["Saturday","Sunday"] else PALETTE[0] for d in dow["day"]]
ax.bar(dow["day"], dow["orders"], color=bar_colors, edgecolor="white", width=0.6)
ax.set_title("Order Volume by Day of Week", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Number of Orders")
ax.spines[["top","right"]].set_visible(False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=PALETTE[0], label="Weekday"),
                   Patch(color=PALETTE[2], label="Weekend")], fontsize=9)
plt.tight_layout()
save("09_day_of_week_orders.png")


# 12. CHART 10 — Discount Impact on Revenue

print("── 12. Chart: Discount Impact ──")

disc = df_completed.copy()
disc["discount_pct"] = (disc["discount"] * 100).astype(int).astype(str) + "%"
disc_group = disc.groupby("discount_pct").agg(
    avg_revenue=("revenue","mean"),
    total_orders=("order_id","count")
).reset_index().sort_values("discount_pct")

fig, ax1 = plt.subplots(figsize=(8, 4.5), facecolor=BG)
ax2 = ax1.twinx()
ax1.set_facecolor(BG)

bars = ax1.bar(disc_group["discount_pct"], disc_group["total_orders"],
               color=PALETTE[0], alpha=0.6, label="Orders", width=0.45)
ax2.plot(disc_group["discount_pct"], disc_group["avg_revenue"],
         marker="D", color=PALETTE[2], linewidth=2.2, markersize=7, label="Avg Revenue")

ax1.set_xlabel("Discount Level")
ax1.set_ylabel("Number of Orders", color=PALETTE[0])
ax2.set_ylabel("Avg Revenue per Order (₹)", color=PALETTE[2])
ax1.tick_params(axis="y", labelcolor=PALETTE[0])
ax2.tick_params(axis="y", labelcolor=PALETTE[2])
ax1.set_title("Discount Level vs Order Volume & Avg Revenue", fontsize=13, fontweight="bold", pad=12)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
ax1.spines[["top","right"]].set_visible(False)
ax2.spines[["top","left"]].set_visible(False)
plt.tight_layout()
save("10_discount_impact.png")


# 13. SUMMARY OUTPUT

print("\n" + "═"*60)
print("  InsightX — Analysis Complete")
print("═"*60)
print(f"  Total orders analysed  : {len(df):,}")
print(f"  Completed revenue      : ₹{df_completed['revenue'].sum():,.0f}")
print(f"  Average order value    : ₹{df_completed['revenue'].mean():,.0f}")
print(f"  Unique customers       : {df['customer_id'].nunique():,}")
print(f"  Top category           : {cat_rev.iloc[-1]['category']}")
print(f"  Charts saved to        : {CHART_DIR}")
print("═"*60)

# Save a summary CSV for Power BI import
summary = df_completed.groupby(["month","category","region","channel"]).agg(
    revenue=("revenue","sum"),
    orders=("order_id","count"),
    avg_discount=("discount","mean"),
    avg_unit_price=("unit_price","mean")
).reset_index().round(2)
summary.to_csv(os.path.join(os.path.dirname(__file__), "data", "summary_for_powerbi.csv"), index=False)
print("  Power BI summary CSV saved → data/summary_for_powerbi.csv")
