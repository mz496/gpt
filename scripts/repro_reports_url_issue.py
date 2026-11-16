import wandb
import wandb_workspaces.reports.v2 as wr
from wandb_workspaces._graphql import get_app_url  # pyright: ignore

ENTITY = wandb.Api().default_entity
PROJECT = "gpt-experiments"

app_url = get_app_url(wandb.Api())
print("app_url:", app_url)

report = wr.Report(entity=ENTITY, project=PROJECT, title="repro-report")
report.save()

# BUILD side: report.url keeps only the scheme and host of app_url, so on Forge the /wandb is missing.
correct_report_url = f"{app_url.rstrip('/')}/{ENTITY}/{PROJECT}/reports/{report.url.rsplit('/', 1)[1]}"
print("report.url:", report.url)
print("expected:  ", correct_report_url)
print("build side matches:", report.url == correct_report_url)

# PARSE side: reading the correct link back splits the path at fixed positions, so the prefix breaks it.
try:
    loaded_report = wr.Report.from_url(correct_report_url)
    print("loaded report:", loaded_report.title)
except Exception as e:
    print(f"Report.from_url(expected) failed: {type(e).__name__}: {e}")
