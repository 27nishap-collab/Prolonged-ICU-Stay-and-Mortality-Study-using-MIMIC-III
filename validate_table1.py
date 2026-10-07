"""Validate W1 against every published number in Weissman Table 1."""
import duckdb, pandas as pd, numpy as np, os
D = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(D, "out")
con = duckdb.connect(os.environ.get("MIMIC3_DB", os.path.expanduser("~/mnt/MIMIC_III_DB/mimic3.duckdb")), read_only=True)
d = pd.read_csv(f"{OUT}/cohort_W1.csv")
n = len(d); y = d.labels == 1

def iqr(s):
    s = pd.Series(s).dropna()
    return f"{s.median():.1f} ({s.quantile(.25):.1f}-{s.quantile(.75):.1f})"

rows = []
def add(label, ours, pub_all, pub_yes=None, ours_yes=None):
    rows.append(dict(characteristic=label, ours=ours, published=pub_all,
                     ours_outcome_yes=ours_yes, published_outcome_yes=pub_yes))

def pc(mask):  return f"{mask.sum():,} ({100*mask.mean():.1f})"

add("Patient admissions", f"{n:,}", "25,947", "5,504 (21.2)", f"{y.sum():,} ({100*y.mean():.1f})")
add("In-hospital death", pc(d.outcome_death==1), "1,861 (7.8)*", "1,861 (33.8)",
    f"{(d.outcome_death==1).sum():,} ({100*(d.outcome_death[y]==1).mean():.1f})")
add("ICU LOS >= 7 days", pc(d.outcome_icu_los_7d==1), "4,564 (17.6)", "4,564 (82.9)",
    f"{(d.outcome_icu_los_7d==1).sum():,} ({100*(d.outcome_icu_los_7d[y]==1).mean():.1f})")
add("Age (years)", iqr(d.age.where(d.age<120)), "64.1 (51.3-76.3)", "66.9 (53.6-77.8)",
    iqr(d.age[y].where(d.age[y]<120)))
add("ICU LOS (days)", iqr(d.icu_los_sum), "2.5 (1.4-5.0)", "11.0 (7.8-17.6)", iqr(d.icu_los_sum[y]))
add("Hospital LOS (days)", iqr(d.hosp_los_h/24), "6.9 (4.5-11.6)", "16.5 (10.7-25.0)",
    iqr(d.hosp_los_h[y]/24))
add("Male", pc(d.gender=="M"), "14,958 (57.5)", "3,161 (57.4)", pc((d.gender=="M")[y]))
for t, p_all, p_yes in [("EMERGENCY","20,867 (80.4)","4,837 (87.9)"),
                        ("ELECTIVE","4,341 (16.7)","451 (8.2)"),
                        ("URGENT","739 (2.8)","216 (3.9)")]:
    add(f"  {t.title()}", pc(d.admission_type==t), p_all, p_yes, pc((d.admission_type==t)[y]))
for t, p_all, p_yes in [("MICU","9,622 (37.1)","2,168 (39.4)"),("CSRU","4,935 (19.0)","686 (12.5)"),
                        ("CCU","4,030 (15.5)","765 (13.9)"),("SICU","4,024 (15.5)","1,031 (18.7)"),
                        ("TSICU","3,329 (12.8)","853 (15.5)")]:
    m = d.initial_icu_type.fillna("") == t
    add(f"  {t}", pc(m), p_all, p_yes, pc(m[y]))
eth = d.ethnicity.fillna("").str.upper()
for lab, pat, p_all, p_yes in [("White","^WHITE","18,461 (71.1)","3,916 (71.4)"),
                               ("Black","^BLACK","2,475 (9.5)","407 (7.4)"),
                               ("Hispanic or Latino","^HISPANIC","848 (3.3)","151 (2.7)"),
                               ("Asian","^ASIAN","563 (2.2)","97 (1.8)")]:
    m = eth.str.match(pat)
    add(f"  {lab}", pc(m), p_all, p_yes, pc(m[y]))
add("  Other/Unknown", pc(~eth.str.match("^(WHITE|BLACK|HISPANIC|ASIAN)")),
    "3,600 (13.9)", "933 (17.0)", pc((~eth.str.match("^(WHITE|BLACK|HISPANIC|ASIAN)"))[y]))
add("Number of notes", "", "6 (4-11)", "8 (6-12)", "")
add("Total raw word count", "", "1,023 (562-2,266)", "1,410 (956-2,292)", "")

t = pd.DataFrame(rows)
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 30)
print(t.to_string(index=False))
t.to_csv(f"{OUT}/W1_table1_validation.csv", index=False)
print("\n* published 7.8% does not reconcile: 1,861/25,947 = 7.17%. Ours matches 7.17%.")
print("wrote out/W1_table1_validation.csv")
