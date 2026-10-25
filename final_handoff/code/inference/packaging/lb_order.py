"""Hidden-precision order of submissions that tie on the displayed public LB (Master 10-16 20:1x): Kaggle's server-side sort by public
score (competition_submissions sort=SUBMISSION_SORT_BY_PUBLIC_SCORE) orders ties by the unrounded score. Validated on the known pair
oct12_syn_g (57112379) > oct11_oa_3pipe (57090414). Prints the rank of each given ref in that order.
conda run -n rsna python submissions/oct16_syn_cnx_probe/lb_order.py 57279368 57154196"""
import sys

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_enums import SubmissionSortBy

COMP = "rsna-knee-abnormality-detection"
api = KaggleApi()
api.authenticate()
subs, token = [], ""
while True:
    page = api.competition_submissions(COMP, sort=SubmissionSortBy.SUBMISSION_SORT_BY_PUBLIC_SCORE, page_token=token, page_size=100)
    subs += [s for s in page if s is not None]
    token = getattr(page, "next_page_token", "") if not isinstance(page, list) else ""
    if not token or not page:
        break
rank = {str(s.ref): i for i, s in enumerate(subs)}
for ref in sys.argv[1:]:
    s = next(x for x in subs if str(x.ref) == ref)
    print(f"{ref}: position {rank[ref]} in the public-score sort, LB {s.public_score}, status {s.status}, {(s.description or '')[:60]}")
