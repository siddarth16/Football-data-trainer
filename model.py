import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("/kaggle/input/enhanced-features-cleaned/enhanced_features_cleaned.csv")
df["Date_parsed"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("Date_parsed")
df["MatchResult"] = df["FTR"].map({"A": 0, "D": 1, "H": 2})
target = "MatchResult"

# Top features
top_features = [
    'Away_LeaguePoints', 'B365H', 'Home_LeaguePoints', 'B365A', 'Rank_Diff',
    'Home_goal_diff_5', 'Home_avg_goals_conceded_5', 'H2H_AvgGoals',
    'Away_goal_diff_5', 'B365D', 'Away_avg_goals_conceded_5', 'Away_Rank',
    'Away_avg_goals_scored_5', 'Home_avg_goals_scored_5', 'Home_RestDays',
    'H2H_WinPct', 'Home_Rank', 'Away_points_per_game_5'
]

# Split train/test
train_df = df[df["Season"] < "2024/2025"]
test_df = df[df["Season"] == "2024/2025"].sort_values("Date_parsed")
X_train, y_train = train_df[top_features], train_df[target]

# Preprocessing
imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train))

# Cross-validation
print("🔍 Running 5-fold CV...")
print("LightGBM AUC:", cross_val_score(LGBMClassifier(), X_train_scaled, y_train, scoring="roc_auc_ovr", cv=5).mean())
print("RandomForest AUC:", cross_val_score(RandomForestClassifier(), X_train_scaled, y_train, scoring="roc_auc_ovr", cv=5).mean())

# Bankroll simulation
results, bankroll_progress = [], []
bankroll, initial_bankroll = 10000, 10000

for date in test_df["Date_parsed"].dropna().sort_values().unique():
    matches_today = test_df[test_df["Date_parsed"] == date]
    if matches_today.empty:
        continue

    # Retrain models daily
    model_lgb = LGBMClassifier(n_estimators=500, learning_rate=0.03, max_depth=6, reg_alpha=0.1, reg_lambda=0.1, random_state=42)
    model_rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
    model_lgb.fit(X_train_scaled, y_train)
    model_rf.fit(X_train_scaled, y_train)

    day_profit = day_investment = 0

    for _, row in matches_today.iterrows():
        try:
            B365H, B365D, B365A = row["B365H"], row["B365D"], row["B365A"]
        except:
            continue

        X_row = row[top_features].to_frame().T
        X_scaled = scaler.transform(imputer.transform(X_row))

        proba_avg = (model_lgb.predict_proba(X_scaled)[0] + model_rf.predict_proba(X_scaled)[0]) / 2
        pred = np.argmax(proba_avg)
        model_prob = proba_avg[pred]
        actual = row[target]

        odds_map = {2: B365H, 1: B365D, 0: B365A}
        odds_used = odds_map[pred]
        market_prob = 1 / odds_used
        value_score = model_prob / market_prob

        if model_prob < 0.55 or value_score < 1.05:
            continue

        # Kelly Criterion
        b = odds_used - 1
        p = model_prob
        q = 1 - p
        kelly_fraction = (b * p - q) / b if b != 0 else 0
        if kelly_fraction <= 0:
            continue

        # Boost draw bets if strong value
        if pred == 1 and value_score > 1.1:
            kelly_fraction *= 1.2

        bet_amount = min(kelly_fraction * bankroll, 0.1 * bankroll)
        profit = bet_amount * odds_used if pred == actual else 0

        day_profit += profit
        day_investment += bet_amount

        results.append({
            "Date": row["Date_parsed"],
            "HomeTeam": row["HomeTeam"],
            "AwayTeam": row["AwayTeam"],
            "Prediction": ["AwayWin", "Draw", "HomeWin"][pred],
            "Actual": ["AwayWin", "Draw", "HomeWin"][actual],
            "Confidence": round(model_prob, 4),
            "ValueScore": round(value_score, 4),
            "OddsUsed": round(odds_used, 2),
            "BetAmount": round(bet_amount, 2),
            "MatchProfit": round(profit, 2)
        })

    bankroll += (day_profit - day_investment)
    bankroll_progress.append({"Date": date, "Bankroll": round(bankroll, 2)})

    # Online learning
    X_train = pd.concat([X_train, matches_today[top_features]], ignore_index=True)
    y_train = pd.concat([y_train, matches_today[target]], ignore_index=True)
    X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train))

# Output summary
results_df = pd.DataFrame(results)
bankroll_df = pd.DataFrame(bankroll_progress)

correct_preds = (results_df["Prediction"] == results_df["Actual"]).sum()
total_bets = len(results_df)
total_invested = results_df["BetAmount"].sum()
total_return = results_df["MatchProfit"].sum()
net_profit = total_return - total_invested

print(f"\n📊 Matches Bet On: {total_bets}")
print(f"✅ Correct Predictions: {correct_preds}")
print(f"💰 Total Invested: ₹{round(total_invested, 2)}")
print(f"💸 Total Return: ₹{round(total_return, 2)}")
print(f"📈 Net Profit/Loss: ₹{round(net_profit, 2)}")
print(f"🏦 Final Bankroll: ₹{round(bankroll, 2)}")

# ROI by odds band
results_df["OddsBand"] = pd.cut(results_df["OddsUsed"], bins=[1, 1.5, 2, 2.5, 3, 3.5],
                                 labels=["1-1.5", "1.5-2", "2-2.5", "2.5-3", "3-3.5"])
roi_summary = results_df.groupby("OddsBand")[["MatchProfit", "BetAmount"]].sum()
roi_summary["ROI"] = (roi_summary["MatchProfit"] / roi_summary["BetAmount"]).round(4)
display(roi_summary)

# Plot bankroll trend
bankroll_df.plot(x="Date", y="Bankroll", title="Bankroll Over Time", figsize=(12, 5))
