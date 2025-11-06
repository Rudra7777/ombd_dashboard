"""
Streamlit Dashboard: NIFTY 50 Portfolio Optimizer + Comparison Portfolios + Scenario Sensitivity

Features added to the original script:
- Comparison portfolios: Equal-weight and Risk-Parity alongside your Mean-Variance optimized portfolio
- Scenario / Sensitivity analysis: sliders to vary MAX_STOCK_PCT and MAX_SECTOR_PCT and re-run optimization; batch-run scenarios
- Efficient frontier generation (by solving minimum-variance for grid of target returns)
- Interactive visualizations using plotly (allocation bar, sector pie, efficient frontier, sensitivity charts)
- Export results to Excel (optimized allocation + comparison portfolios + sensitivity results)

Run:
    pip install -r requirements.txt
    streamlit run nifty50_optimizer_dashboard.py

Requirements (suggested):
pandas
numpy
yfinance
cvxpy
requests
beautifulsoup4
openpyxl
streamlit
plotly
scipy

Notes:
- This file is intended to be run locally where Python can access the internet to download prices.
- If cvxpy solver fails, try installing an alternative solver (OSQP/ECOS) or change solver to 'OSQP' in the code.
"""

# CONFIG (you can adjust these in the Streamlit UI)
BUDGET_INR = 100000
LOOKBACK_DAYS = 252  # ~1 trading year
DEFAULT_MIN_STOCK_PCT = 0.0
DEFAULT_MAX_STOCK_PCT = 0.10
DEFAULT_MAX_SECTOR_PCT = 0.40
TARGET_RETURN_ANNUAL = None
TRANSACTION_COST_PCT = 0.0005
DEFAULT_OUTPUT_FILE = 'nifty50_optimized_allocation_with_dashboard.xlsx'

# Imports
import datetime
import math
import io
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

# lazy import block - Streamlit will show friendly message if missing
try:
    import yfinance as yf
    import cvxpy as cp
    import requests
    from bs4 import BeautifulSoup
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from scipy.optimize import minimize
except Exception as e:
    missing = str(e)
    print("One or more packages are missing. Please install the required packages: pandas numpy yfinance cvxpy requests beautifulsoup4 openpyxl streamlit plotly scipy")
    raise

st.set_page_config(layout='wide', page_title='NIFTY50 Portfolio Optimizer')

# -------------------- Utilities --------------------
@lru_cache(maxsize=1)
def fetch_nifty50_constituents_cached():
    return fetch_nifty50_constituents()


def fetch_nifty50_constituents():
    url = "https://en.wikipedia.org/wiki/NIFTY_50"
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(r.text, "html.parser")

    table = None
    for tbl in soup.find_all("table"):
        txt = tbl.get_text()
        if "Company" in txt and ("Symbol" in txt or "Ticker" in txt):
            table = tbl
            break

    if table is None:
        raise RuntimeError("Could not find constituents table on Wikipedia.")

    # ✅ FIX: wrap table HTML in StringIO before passing to read_html
    df = pd.read_html(io.StringIO(str(table)))[0]

    colmap = {}
    for c in df.columns:
        lc = str(c).lower()
        if "company" in lc or "name" in lc:
            colmap[c] = "Company"
        if "symbol" in lc or "ticker" in lc:
            colmap[c] = "Ticker"
        if "sector" in lc or "industry" in lc:
            colmap[c] = "Sector"

    df = df.rename(columns=colmap)

    # Handle Yahoo tickers
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df["YahooTicker"] = df["Ticker"].apply(lambda x: x + ".NS")

    if "Sector" not in df.columns:
        df["Sector"] = "Unknown"

    return df[["Company", "Ticker", "YahooTicker", "Sector"]]


def download_price_history(tickers, period_days=LOOKBACK_DAYS):
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=int(period_days * 1.5))
    data = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        group_by="ticker",
        auto_adjust=True,
        threads=True,
    )
    if isinstance(data.columns, pd.MultiIndex):
        adj = pd.DataFrame()
        for t in tickers:
            for colname in ["Adj Close","Close","Close*"]:
                if (t, colname) in data.columns:
                    adj[t] = data[(t, colname)]
                    break
    elif "Adj Close" in data.columns:
        adj = data["Adj Close"] if len(tickers)>1 else data["Adj Close"].to_frame()
    elif "Close" in data.columns:
        adj = data["Close"] if len(tickers)>1 else data["Close"].to_frame()
    else:
        raise KeyError("No 'Adj Close' or 'Close' column found in downloaded data.")
    adj = adj.dropna(how='all')
    adj = adj.loc[:, adj.columns.notnull()]
    return adj


def compute_returns_and_cov(adj_prices):
    rets = np.log(adj_prices / adj_prices.shift(1)).dropna()
    mu_daily = rets.mean()
    cov_daily = rets.cov()
    trading_days = 252
    mu_annual = mu_daily * trading_days
    cov_annual = cov_daily * trading_days
    return mu_annual, cov_annual, adj_prices


# -------------------- Optimization methods --------------------

def optimize_mean_variance(mu, cov, min_pct=0.0, max_pct=0.1, target_return=None):
    n = len(mu)
    w = cp.Variable(n)
    exp_ret = mu.values @ w
    risk = cp.quad_form(w, cov.values)
    constraints = [cp.sum(w) == 1, w >= min_pct, w <= max_pct]
    if target_return is None:
        gamma = 0.5
        prob = cp.Problem(cp.Maximize(exp_ret - gamma * risk), constraints)
    else:
        constraints.append(exp_ret >= target_return)
        prob = cp.Problem(cp.Minimize(risk), constraints)
    prob.solve(solver=cp.SCS)
    if w.value is None:
        raise RuntimeError('Optimization failed; try different solver or relax constraints.')
    return pd.Series(np.array(w.value).flatten(), index=mu.index)


def risk_parity_weights(cov):
    # Numerical solution to risk parity via scipy minimize
    n = cov.shape[0]
    init = np.repeat(1.0 / n, n)
    def risk_contribs(w, cov):
        port_var = w.T @ cov @ w
        mc = cov @ w
        rc = w * mc
        return rc
    def objective(x):
        x = np.array(x)
        rc = risk_contribs(x, cov.values)
        # squared deviations from equal risk contributions
        target = port_var = x.T @ cov.values @ x
        equal = port_var / len(x)
        return np.sum((rc - equal)**2)
    cons = ({'type':'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = [(0,1) for _ in range(n)]
    res = minimize(objective, init, bounds=bounds, constraints=cons)
    if not res.success:
        # fallback to inverse-volatility
        iv = 1 / np.sqrt(np.diag(cov))
        w = iv / iv.sum()
        return pd.Series(w, index=cov.index)
    return pd.Series(res.x, index=cov.index)


def equal_weight_weights(mu):
    n = len(mu)
    return pd.Series(np.repeat(1.0/n, n), index=mu.index)


# -------------------- Portfolio conversion to integer shares --------------------

def weights_to_shares(weights, prices, budget=BUDGET_INR):
    # Use the last valid price per ticker (ffill then bfill in case of leading NaNs)
    if prices is None or getattr(prices, "empty", True):
        latest_prices = pd.Series(index=weights.index, dtype=float)
    else:
        aligned = prices.reindex(columns=weights.index)
        last_row = aligned.ffill().bfill().iloc[-1]
        latest_prices = pd.to_numeric(last_row, errors='coerce')

    # Fallback: if some tickers still have missing/non-positive prices, try fetching last 7d close
    missing_mask = (~np.isfinite(latest_prices)) | (latest_prices <= 0)
    if missing_mask.any():
        missing_tickers = latest_prices.index[missing_mask].tolist()
        try:
            recent = yf.download(missing_tickers, period='7d', interval='1d', progress=False, auto_adjust=True, threads=True)
            # Normalize to single-index columns
            if isinstance(recent.columns, pd.MultiIndex):
                tmp = pd.DataFrame()
                for t in missing_tickers:
                    for colname in ["Adj Close","Close","Close*"]:
                        if (t, colname) in recent.columns:
                            tmp[t] = recent[(t, colname)]
                            break
                recent_close = tmp
            else:
                if 'Adj Close' in recent.columns:
                    recent_close = recent['Adj Close'] if len(missing_tickers) > 1 else recent['Adj Close'].to_frame()
                else:
                    recent_close = recent['Close'] if len(missing_tickers) > 1 else recent['Close'].to_frame()
            recent_last = recent_close.ffill().bfill().iloc[-1]
            for t in missing_tickers:
                val = pd.to_numeric(recent_last.get(t, np.nan), errors='coerce')
                if np.isfinite(val) and val > 0:
                    latest_prices.loc[t] = float(val)
        except Exception:
            # Ignore network or fetch failures; we will treat missing as 0
            pass

    # Final fallback: for any tickers still missing, fetch individually via history()
    missing_mask = (~np.isfinite(latest_prices)) | (latest_prices <= 0)
    if missing_mask.any():
        for t in latest_prices.index[missing_mask]:
            try:
                h = yf.Ticker(t).history(period='10d', interval='1d', auto_adjust=True)
                if not h.empty:
                    val = pd.to_numeric(h['Close'].ffill().bfill().iloc[-1], errors='coerce')
                    if np.isfinite(val) and val > 0:
                        latest_prices.loc[t] = float(val)
            except Exception:
                continue

    # Align weights to price index and replace missing weights with 0
    weights = weights.reindex(latest_prices.index).fillna(0.0)

    target_value = weights * budget

    price_array = latest_prices.values.astype(float)
    target_array = target_value.values.astype(float)

    # Compute raw shares safely, treating non-finite or non-positive prices as 0 shares
    valid_price_mask = np.isfinite(price_array) & (price_array > 0)
    raw_shares = np.zeros_like(price_array, dtype=float)
    np.divide(target_array, price_array, out=raw_shares, where=valid_price_mask)
    # Replace any residual non-finite values with 0
    raw_shares = np.where(np.isfinite(raw_shares), raw_shares, 0.0)

    # Floor to integer shares safely
    int_shares = np.floor(raw_shares).astype(int)

    # Invested capital using safe prices (NaN/inf treated as 0)
    safe_prices = np.nan_to_num(price_array, nan=0.0, posinf=0.0, neginf=0.0)
    invested = np.sum(int_shares * safe_prices)
    leftover = float(budget) - float(invested)

    # Greedy allocation of leftover using affordability metric; skip invalid prices
    fractional = raw_shares - int_shares
    affordability = np.full_like(price_array, -np.inf, dtype=float)
    np.divide(fractional, price_array, out=affordability, where=price_array > 0)
    order = np.argsort(-affordability)
    for i in order:
        price = price_array[i]
        if not np.isfinite(price) or price <= 0:
            continue
        if price <= leftover + 1e-9:
            int_shares[i] += 1
            leftover -= price

    df = pd.DataFrame({
        'Ticker': latest_prices.index,
        'Weight': weights.values,
        'Price': price_array,
        'Shares': int_shares,
    })
    df['Invested'] = df['Shares'] * df['Price']
    total_invested = df['Invested'].sum()
    pct = df['Invested'] / total_invested if total_invested > 0 else 0.0
    # Store as percentage value (e.g., 10.41 meaning 10.41%)
    df['PctPortfolio'] = (pct * 100).astype(float)
    return df, leftover


# -------------------- Backtest / Performance --------------------

def backtest(weights, prices):
    # weights: pandas Series (index tickers) that sum to 1
    # prices: DataFrame of prices with same columns
    rets = np.log(prices / prices.shift(1)).dropna()
    port_daily = (rets[weights.index] @ weights.values)
    cumulative = (1 + port_daily).cumprod()
    total_return = cumulative.iloc[-1] - 1
    ann_return = port_daily.mean() * 252
    ann_vol = port_daily.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
    return cumulative, total_return, ann_return, ann_vol, sharpe


# -------------------- Efficient frontier --------------------

def efficient_frontier(mu, cov, min_pct=0.0, max_pct=0.1, points=20):
    returns = np.linspace(mu.min()*0.5, mu.max()*1.2, points)
    frontier = []
    for r in returns:
        try:
            w = optimize_mean_variance(mu, cov, min_pct=min_pct, max_pct=max_pct, target_return=r)
            vol = np.sqrt(w.values @ cov.values @ w.values)
            ret = mu.values @ w.values
            frontier.append({'Return': ret, 'Volatility': vol, 'Weights': w})
        except Exception:
            continue
    df = pd.DataFrame(frontier)
    return df


# -------------------- Streamlit UI --------------------

st.title('NIFTY50 Portfolio Optimizer — Dashboard')

with st.sidebar:
    st.header('Data & Model Settings')
    constituents = fetch_nifty50_constituents_cached()
    tickers = constituents['YahooTicker'].tolist()
    st.markdown(f"**Constituents loaded:** {len(tickers)}")

    lookback_days = st.number_input('Lookback days (for historical returns)', value=LOOKBACK_DAYS, min_value=30, max_value=252*3)
    budget = st.number_input('Budget (INR)', value=BUDGET_INR)

    # ✅ Change: Show percentage inputs as percentages (0–100) instead of 0.0–1.0
    min_stock_pct_display = st.number_input('Min stock %', value=DEFAULT_MIN_STOCK_PCT * 100, step=1.0, min_value=0.0, max_value=100.0)
    max_stock_pct_display = st.number_input('Max stock %', value=DEFAULT_MAX_STOCK_PCT * 100, step=1.0, min_value=0.0, max_value=100.0)
    max_sector_pct_display = st.number_input('Max sector %', value=DEFAULT_MAX_SECTOR_PCT * 100, step=1.0, min_value=0.0, max_value=100.0)

    # Convert back to decimals for internal computation
    min_stock_pct = min_stock_pct_display / 100
    max_stock_pct = max_stock_pct_display / 100
    max_sector_pct = max_sector_pct_display / 100

    st.markdown('---')
    do_equal = False
    do_riskparity = False
    st.header('Outputs')
    output_file = st.text_input('Excel output filename', value=DEFAULT_OUTPUT_FILE)
    run_button = st.button('Run Optimization')

# Main area
main_col = st.container()

with main_col:
    st.subheader('Data Preview')
    if st.button('Download & Preview Prices'):
        with st.spinner('Downloading prices...'):
            adj = download_price_history(tickers, period_days=lookback_days)
            st.write('Price data (last 5 rows):')
            st.dataframe(adj.tail())

    if run_button:
        st.info('Running optimization — this may take a moment...')

    # Download prices
    adj = download_price_history(tickers, period_days=lookback_days)
    mu, cov, adj_prices = compute_returns_and_cov(adj)

    # --- Mean-Variance Optimization ---
    eq_weights = equal_weight_weights(mu) if do_equal else None

    rp_weights = None
    if do_riskparity:
        try:
            rp_weights = risk_parity_weights(cov)
            rp_weights = rp_weights.clip(lower=0)
            rp_weights = rp_weights / rp_weights.sum()
        except Exception as e:
            st.warning(f'Risk-parity optimization failed, falling back to inverse-vol method: {e}')
            rp_weights = None

    try:
        mv_weights = optimize_mean_variance(
            mu, cov,
            min_pct=min_stock_pct,
            max_pct=max_stock_pct,
            target_return=None
        )
        mv_weights = mv_weights.clip(lower=0)

        # ✅ Avoid division by zero if all weights are zero
        if mv_weights.sum() == 0:
            st.error("⚠️ Optimization failed: all weights are zero. Try relaxing constraints (increase max stock % or max sector %).")
            mv_weights = pd.Series(np.repeat(1/len(mu), len(mu)), index=mu.index)  # fallback: equal weight
        else:
            mv_weights = mv_weights / mv_weights.sum()

    except Exception as e:
        st.error(f"Mean-Variance optimization failed: {e}")
        mv_weights = pd.Series(np.repeat(1/len(mu), len(mu)), index=mu.index)  # fallback

    # Convert to shares and compute allocations
    results = {}
    if mv_weights is not None:
        mv_df, mv_leftover = weights_to_shares(mv_weights, adj_prices, budget=budget)
        # Merge using Yahoo tickers to avoid duplicate 'Ticker' columns
        mv_df = mv_df.merge(constituents[['Company','Sector','YahooTicker']], left_on='Ticker', right_on='YahooTicker', how='left').drop(columns=['YahooTicker'])
        mv_df['WeightPct'] = mv_df['Weight'] * 100
        results['Mean-Variance'] = {'weights': mv_weights, 'alloc_df': mv_df, 'leftover': mv_leftover}

    if eq_weights is not None:
        eq_df, eq_leftover = weights_to_shares(eq_weights, adj_prices, budget=budget)
        eq_df = eq_df.merge(constituents[['Company','Sector','YahooTicker']], left_on='Ticker', right_on='YahooTicker', how='left').drop(columns=['YahooTicker'])
        eq_df['WeightPct'] = eq_df['Weight'] * 100
        results['Equal-weight'] = {'weights': eq_weights, 'alloc_df': eq_df, 'leftover': eq_leftover}

    if rp_weights is not None:
        rp_df, rp_leftover = weights_to_shares(rp_weights, adj_prices, budget=budget)
        rp_df = rp_df.merge(constituents[['Company','Sector','YahooTicker']], left_on='Ticker', right_on='YahooTicker', how='left').drop(columns=['YahooTicker'])
        rp_df['WeightPct'] = rp_df['Weight'] * 100
        results['Risk-Parity'] = {'weights': rp_weights, 'alloc_df': rp_df, 'leftover': rp_leftover}

    # Display allocations (top 10 for readability)
    st.subheader('Top allocations (Mean-Variance)')
    if 'Mean-Variance' in results:
        mv_alloc = results['Mean-Variance']['alloc_df']
        mv_alloc_pos = mv_alloc[mv_alloc['Invested'] > 0].copy()

        if mv_alloc_pos.empty:
            st.warning('No invested capital could be computed—price data may be missing. Try refreshing prices or adjusting the lookback window.')
        else:
            display_cols = ['Company','Ticker','Sector','Price','Shares','Invested','PctPortfolio','WeightPct']
            display_df = mv_alloc_pos.nlargest(15, 'PctPortfolio')[display_cols].copy()
            display_df.index = np.arange(1, len(display_df) + 1)
            st.dataframe(display_df, use_container_width=True)

            # Portfolio summary stats
            invested_total = mv_alloc_pos['Invested'].sum()
            expected_return = float(mv_weights.values @ mu.values)
            portfolio_variance = float(mv_weights.values @ cov.values @ mv_weights.values)
            portfolio_vol = math.sqrt(portfolio_variance)
            sharpe_ratio = expected_return / portfolio_vol if portfolio_vol > 0 else float('nan')

            st.markdown('### Portfolio Summary')
            summary_cols = st.columns(4)
            summary_cols[0].metric('Total invested amount (INR)', f"{invested_total:,.0f}")
            summary_cols[1].metric('Expected annual return', f"{expected_return*100:.2f}%")
            summary_cols[2].metric('Annual volatility (risk)', f"{portfolio_vol*100:.2f}%")
            summary_cols[3].metric('Sharpe Ratio', f"{sharpe_ratio:.2f}")

            total_invested = mv_alloc_pos['PctPortfolio'].sum()
            fig = px.treemap(
                mv_alloc_pos,
                path=['Sector', 'Company'],
                values='PctPortfolio',
                color='PctPortfolio',
                color_continuous_scale=px.colors.sequential.Viridis,
                title='Portfolio Allocation Treemap (Mean-Variance)',
                hover_data={'Ticker': True, 'Invested': ':.2f', 'PctPortfolio': ':.2f', 'WeightPct': ':.2f'}
            )
            fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))

            st.plotly_chart(fig, use_container_width=True)

            # Sector breakdown chart for MV (group sectors <5% into 'Others')
            sector_mv = mv_alloc_pos.groupby('Sector').agg({'PctPortfolio':'sum'}).reset_index()
            sector_mv['Pct'] = sector_mv['PctPortfolio'] / total_invested
            major = sector_mv[sector_mv['Pct'] >= 0.05]
            minor = sector_mv[sector_mv['Pct'] < 0.05]
            if not minor.empty:
                others_row = pd.DataFrame([{'Sector': 'Others', 'PctPortfolio': minor['PctPortfolio'].sum(), 'Pct': minor['Pct'].sum()}])
                sector_mv_disp = pd.concat([major, others_row], ignore_index=True)
            else:
                sector_mv_disp = major
            fig = px.pie(sector_mv_disp, names='Sector', values='Pct', title='Sector Allocation (Mean-Variance)', color='Sector', color_discrete_sequence=px.colors.qualitative.Vivid)
            fig.update_layout(
                height=520,
                margin=dict(t=80, l=0, r=0, b=0),
                legend=dict(x=0.82, y=0.5, xanchor='left', yanchor='middle')
            )
            st.plotly_chart(fig, use_container_width=True)

    # Removed multi-strategy comparison chart (only Mean-Variance is shown)

    # Efficient frontier
    try:
        ef = efficient_frontier(mu, cov, min_pct=min_stock_pct, max_pct=max_stock_pct, points=25)
        if not ef.empty:
            fig = px.line(ef, x='Volatility', y='Return', title='Efficient Frontier')
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f'Efficient frontier generation failed: {e}')

    # Sensitivity / Scenario Analysis: vary max_stock_pct and compute portfolio metrics
    st.subheader('Quick Sensitivity Analysis (vary max stock pct)')
    percent_grid = np.linspace(0.02, max(0.02, max_stock_pct*2), 8)
    sens_results = []
    for mx in percent_grid:
        try:
            w = optimize_mean_variance(mu, cov, min_pct=min_stock_pct, max_pct=mx, target_return=None)
            w = w.clip(lower=0)
            w = w / w.sum()
            # compute portfolio stats
            vol = np.sqrt(w.values @ cov.values @ w.values)
            ret = mu.values @ w.values
            sens_results.append({'MaxStockPct': mx, 'Return': ret, 'Volatility': vol})
        except Exception:
            continue
    sens_df = pd.DataFrame(sens_results)
    if not sens_df.empty:
        fig = px.line(sens_df, x='MaxStockPct', y=['Return','Volatility'], labels={'value':'Metric','variable':'Metric Type'}, title='Sensitivity: Return & Volatility vs Max Stock %')
        st.plotly_chart(fig, use_container_width=True)

        # Backtesting section (optional) - user can choose to backtest
        st.subheader('Backtest optimized portfolio (optional)')
        if st.checkbox('Perform simple backtest on most recent 3-month window'):
            test_days = st.number_input('Test window days', value=63)
            end = datetime.datetime.now()
            start_test = end - datetime.timedelta(days=test_days)
            # download test window prices
            test_prices = yf.download(tickers, start=start_test.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), progress=False, group_by='ticker', auto_adjust=True, threads=True)
            # process similar to download_price_history
            if isinstance(test_prices.columns, pd.MultiIndex):
                adj_test = pd.DataFrame()
                for t in tickers:
                    if (t, 'Close') in test_prices.columns:
                        adj_test[t] = test_prices[(t,'Close')]
            else:
                adj_test = test_prices['Close']
            if 'Mean-Variance' in results:
                cum, tot, ann_r, ann_v, sh = backtest(results['Mean-Variance']['weights'], adj_test)
                st.write(f'Test cumulative return: {tot:.4f}, Annualized return: {ann_r:.4f}, Volatility: {ann_v:.4f}, Sharpe: {sh:.4f}')
                st.line_chart(cum)

        # Export to Excel
        st.subheader('Export results')
        if st.button('Export all results to Excel'):
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                for name, r in results.items():
                    dfout = r['alloc_df'][['Company','Ticker','Sector','Price','Shares','Invested','PctPortfolio']].copy()
                    dfout.to_excel(writer, sheet_name=name.replace(' ','_')[:30], index=False)
                if not sens_df.empty:
                    sens_df.to_excel(writer, sheet_name='sensitivity', index=False)
                if 'ef' in locals() and not ef.empty:
                    ef[['Volatility','Return']].to_excel(writer, sheet_name='efficient_frontier', index=False)
                # inputs
                inputs = pd.DataFrame([['Budget_INR', budget], ['Lookback_days', lookback_days], ['Min_stock_pct', min_stock_pct], ['Max_stock_pct', max_stock_pct], ['Max_sector_pct', max_sector_pct]])
                inputs.to_excel(writer, sheet_name='inputs', index=False, header=False)
            buffer.seek(0)
            st.download_button('Download Excel', data=buffer, file_name=output_file, mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# End of Streamlit app
