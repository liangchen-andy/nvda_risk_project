"""All the general configuration of the project."""

from pathlib import Path

SRC: Path = Path(__file__).parent.resolve()
ROOT: Path = SRC.joinpath("..", "..").resolve()

BLD: Path = ROOT.joinpath("bld").resolve()
DATA: Path = BLD.joinpath("data").resolve()
RAW_DATA: Path = DATA.joinpath("raw").resolve()
INTERIM_DATA: Path = DATA.joinpath("interim").resolve()
PROCESSED_DATA: Path = DATA.joinpath("processed").resolve()
CHECKS_OUTPUT: Path = BLD.joinpath("checks").resolve()
REPO_DATA: Path = SRC.joinpath("data").resolve()
SNAPSHOTS_DATA: Path = REPO_DATA.joinpath("snapshots").resolve()

# Canonical output locations for the NVDA risk workflow.
CLEAN_DATA: Path = DATA.joinpath("clean").resolve()
METRICS_OUTPUT: Path = BLD.joinpath("metrics").resolve()
TABLES_OUTPUT: Path = BLD.joinpath("tables").resolve()
FIGURES_OUTPUT: Path = BLD.joinpath("figures").resolve()
PANEL_DAILY_DATA: Path = CLEAN_DATA.joinpath("panel_daily.parquet").resolve()
DATA_PROVENANCE: Path = CLEAN_DATA.joinpath("data_provenance.json").resolve()

# Backward-compatible aliases for existing task modules.
ANALYSIS_OUTPUT: Path = METRICS_OUTPUT


DOCUMENTS: Path = ROOT.joinpath("documents").resolve()
DOCUMENTS_TABLES: Path = DOCUMENTS.joinpath("tables").resolve()
DOCUMENTS_PUBLIC: Path = DOCUMENTS.joinpath("public").resolve()

SUMMARY_TABLE: Path = DOCUMENTS_TABLES.joinpath("summary_all.md").resolve()
DIAGNOSTICS_TABLE: Path = DOCUMENTS_TABLES.joinpath("diagnostics.md").resolve()
SUMMARY_FIG_VOLATILITY: Path = DOCUMENTS_PUBLIC.joinpath("fig_volatility.png").resolve()
SUMMARY_FIG_VAR_EXCEEDANCES: Path = DOCUMENTS_PUBLIC.joinpath(
    "fig_var_exceedances.png",
).resolve()
SUMMARY_FIG_DRAWDOWN: Path = DOCUMENTS_PUBLIC.joinpath("fig_drawdown.png").resolve()
SUMMARY_FIG_BETA_ROLLING: Path = DOCUMENTS_PUBLIC.joinpath("fig_beta_rolling.png").resolve()

DIAGNOSTICS_DATA: Path = METRICS_OUTPUT.joinpath("diagnostics.csv").resolve()
SNAPSHOT_NVDA_DAILY: Path = SNAPSHOTS_DATA.joinpath("nvda_daily.csv").resolve()
SNAPSHOT_MARKET_DAILY: Path = SNAPSHOTS_DATA.joinpath("sp500_daily.csv").resolve()

ASSET_TICKER: str = "NVDA"
BENCHMARK_TICKER: str = "^GSPC"
SAMPLE_START_DATE: str = "2020-01-01"
SAMPLE_END_DATE: str = "2024-12-31"
VAR_LEVEL: float = 0.05
VOL_WINDOW_DAYS: int = 252
LIQUIDITY_WINDOW_DAYS: int = 21
RISK_DIMENSIONS: tuple[str, ...] = (
    "market",
    "liquidity",
    "drawdown",
    "systematic",
    "macro",
)
