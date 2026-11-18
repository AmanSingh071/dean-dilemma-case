import io, os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    balanced_accuracy_score, matthews_corrcoef)
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import skew

st.set_page_config(page_title="Dean's Dilemma | Analytics Lab", page_icon="🎓", layout="wide")

NUM_DEFAULT = ["Percent_SSC", "Percent_HSC", "Percent_Degree", "Experience_Yrs", "Percentile_ET", "Percent_MBA", "S-TEST*SCORE", "Marks_Communication", "Marks_Projectwork", "Marks_BOCA"]
CAT_DEFAULT = ["Gender", "Board_SSC", "Board_HSC", "Stream_HSC", "Course_Degree", "Entrance_Test", "Specialization_MBA"]
TARGET_DEFAULT = "Placement_B"
RANDOM_STATE = 42

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: #f7f9fc; border: 1px solid #e6eaf0; padding: 14px; border-radius: 12px;}
.hero {padding: 1.3rem 1.5rem; border-radius: 16px; background: linear-gradient(115deg,#10243a,#1f5672); color:white; margin-bottom:1rem;}
.hero h1 {color:white; margin:0;} .hero p {color:#d9e8f2; margin:.35rem 0 0 0;}
.small-note {color:#667085;font-size:.88rem;}
</style>
<div class="hero"><h1>🎓 Dean's Dilemma | Placement Analytics Lab</h1><p>Explore placement outcomes, CCA, data quality, predictive models and synthetic-data comparisons — locally, using your Excel dataset.</p></div>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_excel(file_bytes, file_name):
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    sheet = "Placement" if "Placement" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)
    df.columns = df.columns.astype(str).str.strip()
    df = df.replace(r"^\s*$", np.nan, regex=True)
    return df, sheet


def make_ohe():
    try: return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError: return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(num_cols, cat_cols, imputation="Median"):
    transformers=[]
    if num_cols:
        if imputation == "MICE (Iterative Imputer)":
            try:
                from sklearn.experimental import enable_iterative_imputer  # noqa
                from sklearn.impute import IterativeImputer
                num_imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10,
                    initial_strategy="median", random_state=RANDOM_STATE, skip_complete=True)
            except Exception as e:
                st.warning(f"MICE unavailable ({e}); median imputation will be used.")
                num_imputer = SimpleImputer(strategy="median")
        else:
            num_imputer = SimpleImputer(strategy="median")
        num_pipe=Pipeline([("imputer",num_imputer),("scaler",StandardScaler())])
        transformers.append(("numeric",num_pipe,num_cols))
    if cat_cols:
        cat_pipe=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",make_ohe())])
        transformers.append(("categorical",cat_pipe,cat_cols))
    return ColumnTransformer(transformers=transformers,remainder="drop")


def make_models(seed=42):
    ada_params = dict(random_state=seed, n_estimators=150, learning_rate=0.05)
    try:
        ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=seed), **ada_params)
    except TypeError:
        ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, random_state=seed), **ada_params)
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000,class_weight="balanced",random_state=seed),
        "LDA": LinearDiscriminantAnalysis(),
        "Random Forest": RandomForestClassifier(n_estimators=400,min_samples_leaf=2,class_weight="balanced",random_state=seed,n_jobs=-1),
        "AdaBoost": ada,
    }


def evaluate(name, model, X, y, threshold=0.5):
    probs=model.predict_proba(X)[:,1]
    pred=(probs>=threshold).astype(int)
    return {"Model":name,"Accuracy":accuracy_score(y,pred),"Balanced Accuracy":balanced_accuracy_score(y,pred),
        "Precision (Placed)":precision_score(y,pred,zero_division=0),"Recall (Placed)":recall_score(y,pred,zero_division=0),
        "F1 (Placed)":f1_score(y,pred,zero_division=0),"ROC-AUC":roc_auc_score(y,probs),
        "MCC":matthews_corrcoef(y,pred)}


def fig_download(fig):
    buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=180,bbox_inches="tight"); buf.seek(0); return buf

# Sidebar: data source
with st.sidebar:
    st.header("⚙️ Workspace")
    default_path=os.path.join(os.path.dirname(__file__),"Dean Dilemma.xlsx")
    uploaded=st.file_uploader("Upload Excel workbook",type=["xlsx","xls"])
    if uploaded is not None:
        raw, sheet_name=load_excel(uploaded.getvalue(),uploaded.name)
        st.success(f"Loaded: {uploaded.name}")
    elif os.path.exists(default_path):
        with open(default_path,"rb") as f: raw,sheet_name=load_excel(f.read(),"Dean Dilemma.xlsx")
        st.info("Using bundled Dean Dilemma.xlsx")
    else:
        st.warning("Upload your Excel workbook to begin.")
        st.stop()
    st.caption(f"Sheet: {sheet_name} · {raw.shape[0]} rows × {raw.shape[1]} columns")
    target=st.selectbox("Target column", options=list(raw.columns), index=list(raw.columns).index(TARGET_DEFAULT) if TARGET_DEFAULT in raw.columns else 0)
    seed=st.number_input("Random seed",min_value=0,max_value=99999,value=42,step=1)

# Data cleanup and feature selection
raw=raw.copy()
for c in raw.columns:
    if raw[c].dtype == "object": raw[c]=raw[c].replace(r"^\s*$",np.nan,regex=True)
if target not in raw: st.error("Target column not found."); st.stop()
raw=raw.dropna(subset=[target]).copy()
# Target encode; binary target must be 2 classes
vals=list(pd.Series(raw[target].dropna().unique()))
if len(vals)!=2:
    st.error(f"The selected target must have exactly 2 classes. Found: {vals}"); st.stop()
if set(pd.to_numeric(pd.Series(vals),errors="coerce").dropna().tolist())=={0,1}:
    raw[target]=pd.to_numeric(raw[target]).astype(int); target_labels={0:"Not placed",1:"Placed"}
else:
    vals_sorted=sorted([str(v) for v in vals])
    mapping={vals_sorted[0]:0,vals_sorted[1]:1}
    raw[target]=raw[target].astype(str).map(mapping).astype(int)
    target_labels={0:vals_sorted[0],1:vals_sorted[1]}

exclude=set([target,"SlNo","cust_id","Placement","Salary"])
# Use notebook's preferred features where available, and avoid engineered one-hot duplicates.
preferred=[c for c in NUM_DEFAULT+CAT_DEFAULT if c in raw.columns and c not in exclude]
if preferred:
    feature_cols=preferred
else:
    feature_cols=[c for c in raw.columns if c not in exclude and raw[c].nunique(dropna=True)>1]
# Detect actual numeric/categorical types; numeric-looking object columns are retained as categorical unless mostly numeric.
for c in feature_cols:
    if raw[c].dtype=="object":
        conv=pd.to_numeric(raw[c],errors="coerce")
        if conv.notna().mean()>.90: raw[c]=conv
num_cols=[c for c in feature_cols if pd.api.types.is_numeric_dtype(raw[c])]
cat_cols=[c for c in feature_cols if c not in num_cols]

# Sidebar analysis controls
with st.sidebar:
    st.divider(); st.subheader("Analysis settings")
    imputation=st.selectbox("Numerical missing-value method",["Median","MICE (Iterative Imputer)"],index=0,
        help="MICE uses chained iterative regression for numerical predictors. Categorical variables use mode imputation.")
    test_size=st.slider("Test set size",0.15,0.40,0.20,0.05)
    threshold=st.slider("Classification threshold",0.10,0.90,0.50,0.05)
    run_cv=st.checkbox("Run 5-fold cross-validation",value=True)
    run_tuning=st.checkbox("Run hyperparameter tuning (slower)",value=False)

# Main tabs
tabs=st.tabs(["Executive Dashboard","Data Quality & CCA","EDA & Relationships","Classification","Synthetic Data & Comparison","Student Risk Scoring","Downloads & Methodology"])

# Shared split/model state: compute on demand and cache per current settings
@st.cache_data(show_spinner=False)
def split_data(data, features, target_col, test_fraction, random_seed):
    X=data[features].copy(); y=data[target_col].astype(int).copy()
    return train_test_split(X,y,test_size=test_fraction,random_state=random_seed,stratify=y)

@st.cache_resource(show_spinner="Training classification models...")
def train_all(data, features, nums, cats, target_col, test_fraction, random_seed, imp, threshold_value, do_cv, do_tuning):
    X_train,X_test,y_train,y_test=split_data(data,features,target_col,test_fraction,random_seed)
    models=make_models(random_seed); fitted={}; rows=[]; cms={}; probs={}; preds={}; reports={}
    for name,est in models.items():
        pipe=Pipeline([("preprocess",make_preprocessor(nums,cats,imp)),("model",est)])
        pipe.fit(X_train,y_train); fitted[name]=pipe
        p=pipe.predict_proba(X_test)[:,1]; pr=(p>=threshold_value).astype(int)
        probs[name]=p; preds[name]=pr; cms[name]=confusion_matrix(y_test,pr,labels=[0,1])
        rows.append(evaluate(name,pipe,X_test,y_test,threshold_value))
        reports[name]=classification_report(y_test,pr,output_dict=True,zero_division=0)
    results=pd.DataFrame(rows).sort_values("ROC-AUC",ascending=False)
    cv_df=None
    if do_cv:
        cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=random_seed); cr=[]
        for name,est in models.items():
            pipe=Pipeline([("preprocess",make_preprocessor(nums,cats,imp)),("model",est)])
            scores=cross_validate(pipe,X_train,y_train,cv=cv,scoring=["accuracy","precision","recall","f1","roc_auc"],n_jobs=-1)
            cr.append({"Model":name,"CV Accuracy":scores["test_accuracy"].mean(),"CV Accuracy SD":scores["test_accuracy"].std(),
                "CV Precision":scores["test_precision"].mean(),"CV Recall":scores["test_recall"].mean(),"CV F1":scores["test_f1"].mean(),"CV ROC-AUC":scores["test_roc_auc"].mean()})
        cv_df=pd.DataFrame(cr).sort_values("CV ROC-AUC",ascending=False)
    tuned={}
    if do_tuning:
        rf_pipe=Pipeline([("preprocess",make_preprocessor(nums,cats,imp)),("model",RandomForestClassifier(random_state=random_seed,n_jobs=-1,class_weight="balanced"))])
        rf_params={"model__n_estimators":[100,200,300,500],"model__max_depth":[None,3,5,8,12],"model__min_samples_split":[2,5,10],"model__min_samples_leaf":[1,2,4],"model__max_features":["sqrt","log2"]}
        rf_search=RandomizedSearchCV(rf_pipe,rf_params,n_iter=18,cv=StratifiedKFold(5,shuffle=True,random_state=random_seed),scoring="roc_auc",random_state=random_seed,n_jobs=-1)
        rf_search.fit(X_train,y_train); tuned["Tuned Random Forest"]={"best_params":rf_search.best_params_,"cv_auc":rf_search.best_score_,"model":rf_search.best_estimator_}
        tuned_model=rf_search.best_estimator_; p=tuned_model.predict_proba(X_test)[:,1]; pr=(p>=threshold_value).astype(int)
        rows2=evaluate("Tuned Random Forest",tuned_model,X_test,y_test,threshold_value)
        results=pd.concat([results,pd.DataFrame([rows2])],ignore_index=True)
        fitted["Tuned Random Forest"]=tuned_model; probs["Tuned Random Forest"]=p; preds["Tuned Random Forest"]=pr; cms["Tuned Random Forest"]=confusion_matrix(y_test,pr,labels=[0,1])
        tuned["Tuned Random Forest"]["model"]=tuned_model
    return {"X_train":X_train,"X_test":X_test,"y_train":y_train,"y_test":y_test,"models":fitted,"results":results,"cv":cv_df,"cms":cms,"probs":probs,"preds":preds,"reports":reports,"tuned":tuned}

# Tab 1 Executive
with tabs[0]:
    st.subheader("Placement outcome at a glance")
    placed=int((raw[target]==1).sum()); notplaced=int((raw[target]==0).sum()); n=len(raw)
    a,b,c,d=st.columns(4)
    a.metric("Students",f"{n:,}"); b.metric("Placed",f"{placed:,}",f"{placed/n:.1%}"); c.metric("Not placed",f"{notplaced:,}",f"{notplaced/n:.1%}"); d.metric("Predictors",len(feature_cols))
    l,r=st.columns([1,1])
    with l:
        st.markdown("#### Placement distribution")
        fig,ax=plt.subplots(figsize=(6,4)); counts=raw[target].map(target_labels).value_counts(); counts.plot(kind="bar",ax=ax); ax.set_ylabel("Students"); ax.set_xlabel(""); ax.tick_params(axis='x',rotation=0); st.pyplot(fig); plt.close(fig)
    with r:
        st.markdown("#### Academic profile — mean values")
        if num_cols:
            means=raw[num_cols].mean(numeric_only=True).sort_values(ascending=False)
            st.dataframe(means.rename("Mean").to_frame().round(2),use_container_width=True)
    st.info("Interpretation: because the outcome is imbalanced, accuracy alone can be misleading. Always inspect recall for the not-placed class and the confusion matrix.")
    st.markdown("#### Data snapshot")
    st.dataframe(raw.head(10),use_container_width=True)

# Tab 2 Data quality
with tabs[1]:
    st.subheader("Data quality and missing-value diagnostics")
    miss=pd.DataFrame({"Missing Count":raw[feature_cols].isna().sum(),"Missing %":raw[feature_cols].isna().mean()*100}).sort_values("Missing Count",ascending=False)
    total_cells=raw[feature_cols].size; missing_cells=int(raw[feature_cols].isna().sum().sum()); incomplete=int(raw[feature_cols+[target]].isna().any(axis=1).sum())
    m1,m2,m3=st.columns(3); m1.metric("Missing cells",f"{missing_cells:,}"); m2.metric("Incomplete rows",f"{incomplete:,}"); m3.metric("Complete-case retention",f"{(n-incomplete)/n:.1%}")
    st.markdown("#### Missing values by variable"); st.dataframe(miss.round(2),use_container_width=True)
    fig,ax=plt.subplots(figsize=(9,4)); miss["Missing Count"].sort_values().plot(kind="barh",ax=ax); ax.set_xlabel("Missing cells"); st.pyplot(fig); plt.close(fig)
    cca=raw[feature_cols+[target]].dropna(); st.markdown("#### Complete Case Analysis (CCA)")
    st.write(f"Original records: **{len(raw)}** · Complete cases: **{len(cca)}** · Records excluded by CCA: **{len(raw)-len(cca)}**")
    st.caption("CCA is shown as a diagnostic. Model pipelines impute within each training fold; records are not automatically deleted.")
    if num_cols:
        st.markdown("#### Original vs CCA: numerical distribution comparison")
        cca_num = st.selectbox("Numerical variable for CCA comparison", num_cols, key="cca_num")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.kdeplot(raw[cca_num].dropna(), label="Original", ax=ax, fill=False)
        if not cca.empty and cca[cca_num].nunique() > 1:
            sns.kdeplot(cca[cca_num].dropna(), label="Complete cases (CCA)", ax=ax, fill=False)
        ax.set_title(f"Original vs CCA — {cca_num}"); ax.legend(); st.pyplot(fig); plt.close(fig)
    if cat_cols:
        st.markdown("#### Original vs CCA: categorical composition")
        cca_cat = st.selectbox("Categorical variable for CCA comparison", cat_cols, key="cca_cat")
        original_dist = raw[cca_cat].value_counts(normalize=True, dropna=False).rename("Original")
        cca_dist = cca[cca_cat].value_counts(normalize=True, dropna=False).rename("CCA")
        compare_dist = pd.concat([original_dist, cca_dist], axis=1).fillna(0)
        st.dataframe((compare_dist * 100).round(2), use_container_width=True)
        fig, ax = plt.subplots(figsize=(8, 4)); (compare_dist * 100).plot(kind="bar", ax=ax)
        ax.set_ylabel("Share (%)"); ax.set_title(f"Original vs CCA — {cca_cat}"); ax.tick_params(axis="x", rotation=35)
        st.pyplot(fig); plt.close(fig)
    st.markdown("#### Outlier diagnostics")
    flags=pd.DataFrame(index=raw.index)
    for col in num_cols:
        s=pd.to_numeric(raw[col],errors="coerce"); q1,q3=s.quantile([.25,.75]); iqr=q3-q1
        flags[col+"_IQR"]=(s<q1-1.5*iqr)|(s>q3+1.5*iqr)
        sd=s.std(); flags[col+"_Z"]=(np.abs((s-s.mean())/sd)>3) if sd and np.isfinite(sd) else False
        lo,hi=s.quantile([.01,.99]); flags[col+"_Percentile"]=(s<lo)|(s>hi)
    if num_cols:
        imp=SimpleImputer(strategy="median"); z=StandardScaler().fit_transform(imp.fit_transform(raw[num_cols]))
        flags["Isolation Forest"]=IsolationForest(n_estimators=200,contamination=.03,random_state=int(seed)).fit_predict(z)==-1
    flags=flags.fillna(False).astype(bool)
    method_summary=pd.DataFrame({"Method":["IQR","Z-score","1st/99th percentile","Isolation Forest"],"Rows flagged":[int(flags[[c for c in flags if c.endswith('_IQR')]].any(axis=1).sum()),int(flags[[c for c in flags if c.endswith('_Z')]].any(axis=1).sum()),int(flags[[c for c in flags if c.endswith('_Percentile')]].any(axis=1).sum()),int(flags["Isolation Forest"].sum()) if "Isolation Forest" in flags else 0]})
    st.dataframe(method_summary,use_container_width=True)
    st.metric("Rows flagged by at least 2 methods",int((method_summary.set_index("Method").loc["IQR","Rows flagged"]>0) and 0) if False else int((flags.sum(axis=1)>=2).sum()))
    st.markdown("#### Skewness"); sk=pd.DataFrame({"Variable":num_cols,"Skewness":[raw[c].skew() for c in num_cols]}); st.dataframe(sk.sort_values("Skewness",key=lambda s:s.abs(),ascending=False).round(3),use_container_width=True)

# Tab 3 EDA
with tabs[2]:
    st.subheader("Exploratory analysis")
    if num_cols:
        col=st.selectbox("Choose numerical variable",num_cols)
        f1,f2=st.columns(2)
        with f1:
            fig,ax=plt.subplots(figsize=(6,4)); sns.histplot(raw[col].dropna(),kde=True,ax=ax); ax.set_title(f"Distribution: {col}"); st.pyplot(fig); plt.close(fig)
        with f2:
            fig,ax=plt.subplots(figsize=(6,4)); sns.boxplot(x=raw[col],ax=ax); ax.set_title(f"Outlier view: {col}"); st.pyplot(fig); plt.close(fig)
    if num_cols:
        st.markdown("#### Correlation matrix")
        corr=raw[num_cols].corr(numeric_only=True)
        fig,ax=plt.subplots(figsize=(10,7)); sns.heatmap(corr,annot=True,fmt=".2f",cmap="coolwarm",center=0,ax=ax); st.pyplot(fig); plt.close(fig)
        st.markdown("#### Relationship with placement")
        means=raw.groupby(target)[num_cols].mean().T.rename(columns=target_labels)
        st.dataframe(means.round(2),use_container_width=True)
        fig,ax=plt.subplots(figsize=(10,5)); means.plot(kind="bar",ax=ax); ax.set_ylabel("Mean"); ax.tick_params(axis='x',rotation=45); st.pyplot(fig); plt.close(fig)
    if cat_cols:
        cat=st.selectbox("Categorical variable",cat_cols)
        tab=pd.crosstab(raw[cat],raw[target].map(target_labels),normalize="index")*100
        st.markdown(f"#### Placement composition by {cat}"); st.dataframe(tab.round(1),use_container_width=True)
        fig,ax=plt.subplots(figsize=(8,4)); tab.plot(kind="bar",stacked=True,ax=ax); ax.set_ylabel("Percent"); ax.tick_params(axis='x',rotation=35); st.pyplot(fig); plt.close(fig)

# Tab 4 Classification
with tabs[3]:
    st.subheader("Classification model comparison")
    st.caption("Models follow the cleaned notebook's setup: stratified train/test split, median or MICE numerical imputation, mode categorical imputation, one-hot encoding, scaling and class weighting where specified.")
    if st.button("🚀 Train / refresh all models",type="primary",key="train_models"):
        st.session_state["model_bundle"]=train_all(raw,feature_cols,num_cols,cat_cols,target,float(test_size),int(seed),imputation,float(threshold),run_cv,run_tuning)
    bundle=st.session_state.get("model_bundle")
    if bundle:
        res=bundle["results"].copy(); st.markdown("#### Test-set performance"); st.dataframe(res.style.format({c:"{:.3f}" for c in res.columns if c!="Model"}),use_container_width=True)
        best=res.sort_values("ROC-AUC",ascending=False).iloc[0]
        q1,q2,q3=st.columns(3); q1.metric("Highest test ROC-AUC",best["Model"],f"{best['ROC-AUC']:.3f}"); q2.metric("Best model accuracy",f"{best['Accuracy']:.1%}"); q3.metric("Best model balanced accuracy",f"{best['Balanced Accuracy']:.1%}")
        fig,ax=plt.subplots(figsize=(10,4)); res.set_index("Model")[["Accuracy","Balanced Accuracy","F1 (Placed)","ROC-AUC"]].plot(kind="bar",ax=ax); ax.set_ylim(0,1); ax.tick_params(axis='x',rotation=25); st.pyplot(fig); plt.close(fig)
        if bundle["cv"] is not None:
            st.markdown("#### 5-fold cross-validation (training data only)"); st.dataframe(bundle["cv"].round(4),use_container_width=True)
        if bundle["tuned"]:
            st.markdown("#### Tuned Random Forest"); st.write("Best CV ROC-AUC:",round(bundle["tuned"]["Tuned Random Forest"]["cv_auc"],4)); st.json(bundle["tuned"]["Tuned Random Forest"]["best_params"])
        chosen=st.selectbox("Inspect model",list(bundle["models"].keys()))
        cm=bundle["cms"][chosen]; y_test=bundle["y_test"]; p=bundle["probs"][chosen]; pred=bundle["preds"][chosen]
        st.markdown(f"#### {chosen}: confusion matrix")
        fig,ax=plt.subplots(figsize=(5,4)); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=[target_labels[0],target_labels[1]],yticklabels=[target_labels[0],target_labels[1]],ax=ax); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); st.pyplot(fig); plt.close(fig)
        st.markdown("#### Class-level report"); st.dataframe(pd.DataFrame(classification_report(y_test,pred,output_dict=True,zero_division=0)).T.round(3),use_container_width=True)
        fpr,tpr,_=roc_curve(y_test,p); fig,ax=plt.subplots(figsize=(6,4)); ax.plot(fpr,tpr,label=f"AUC={roc_auc_score(y_test,p):.3f}"); ax.plot([0,1],[0,1],"--"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(); ax.set_title("ROC curve"); st.pyplot(fig); plt.close(fig)
        prec,rec,_=precision_recall_curve(y_test,p); fig,ax=plt.subplots(figsize=(6,4)); ax.plot(rec,prec); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision–Recall curve"); st.pyplot(fig); plt.close(fig)
        st.markdown("#### Threshold analysis")
        threshold_rows=[]
        for t in np.arange(.1,.91,.05):
            pr=(p>=t).astype(int); threshold_rows.append({"Threshold":round(float(t),2),"Accuracy":accuracy_score(y_test,pr),"Recall – class 1":recall_score(y_test,pr,zero_division=0),"Precision – class 1":precision_score(y_test,pr,zero_division=0),"F1 – class 1":f1_score(y_test,pr,zero_division=0),"Recall – class 0":recall_score(1-y_test,1-pr,zero_division=0)})
        st.dataframe(pd.DataFrame(threshold_rows).round(3),use_container_width=True)
        if "Random Forest" in bundle["models"]:
            model=bundle["models"]["Random Forest"]
            try:
                names=model.named_steps["preprocess"].get_feature_names_out(); imp=model.named_steps["model"].feature_importances_
                imp_df=pd.DataFrame({"Feature":names,"Importance":imp}).sort_values("Importance",ascending=False).head(20)
                st.markdown("#### Random Forest feature importance"); st.dataframe(imp_df,use_container_width=True)
                fig,ax=plt.subplots(figsize=(8,6)); sns.barplot(data=imp_df,y="Feature",x="Importance",ax=ax); st.pyplot(fig); plt.close(fig)
            except Exception as e: st.caption(f"Feature importance unavailable: {e}")
        st.download_button("Download model comparison CSV",res.to_csv(index=False).encode(),"model_comparison.csv","text/csv")
    else:
        st.info("Click **Train / refresh all models** to generate test metrics, cross-validation, confusion matrices, ROC/PR curves, threshold analysis and feature importance.")

# Tab 6 synthetic
with tabs[4]:
    st.subheader("Synthetic data generation and comparison")
    st.write("The notebook used SDV's Gaussian Copula synthesizer. This section optionally reproduces that workflow if SDV is installed in the local environment.")
    st.warning("Synthetic data should be assessed for fidelity and privacy. A larger row count does not create new independent information, and training/test splits must be designed to avoid leakage.")
    syn_n=st.number_input("Synthetic records to generate",min_value=100,max_value=100000,value=10000,step=100)
    if st.button("Generate synthetic data with SDV",type="primary"):
        try:
            from sdv.metadata import Metadata
            from sdv.single_table import GaussianCopulaSynthesizer
            syn_source=raw.drop(columns=[c for c in ["SlNo"] if c in raw.columns]).copy()
            meta=Metadata.detect_from_dataframe(data=syn_source)
            synth=GaussianCopulaSynthesizer(meta,enforce_min_max_values=True,enforce_rounding=True)
            with st.spinner("Fitting Gaussian Copula and sampling..."):
                synth.fit(syn_source); syn=synth.sample(num_rows=int(syn_n))
            st.session_state["synthetic_df"]=syn
            st.success(f"Generated {len(syn):,} synthetic rows.")
        except ImportError:
            st.error("SDV is not installed. Run: pip install sdv")
        except Exception as e: st.error(f"Synthetic generation failed: {e}")
    if "synthetic_df" in st.session_state:
        syn=st.session_state["synthetic_df"]
        st.dataframe(syn.head(20),use_container_width=True)
        if target in syn:
            st.markdown("#### Target distribution: original vs synthetic")
            comp=pd.DataFrame({"Original":raw[target].value_counts(normalize=True),"Synthetic":syn[target].value_counts(normalize=True)}).rename(index=target_labels)
            st.dataframe((comp*100).round(1),use_container_width=True)
        st.markdown("#### Original vs synthetic: numerical fidelity")
        shared_num = [c for c in num_cols if c in syn.columns]
        if shared_num:
            selected_syn_num = st.selectbox("Numerical variable", shared_num, key="syn_num")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.kdeplot(raw[selected_syn_num].dropna(), label="Original", ax=ax)
            sns.kdeplot(syn[selected_syn_num].dropna(), label="Synthetic", ax=ax)
            ax.set_title(f"Original vs synthetic — {selected_syn_num}"); ax.legend(); st.pyplot(fig); plt.close(fig)
            fidelity_rows=[]
            for c in shared_num:
                a=pd.to_numeric(raw[c],errors="coerce").dropna(); b=pd.to_numeric(syn[c],errors="coerce").dropna()
                fidelity_rows.append({"Variable":c,"Original mean":a.mean(),"Synthetic mean":b.mean(),"Mean difference":b.mean()-a.mean(),"Original SD":a.std(),"Synthetic SD":b.std()})
            st.dataframe(pd.DataFrame(fidelity_rows).round(3),use_container_width=True)
        shared_cat = [c for c in cat_cols if c in syn.columns]
        if shared_cat:
            st.markdown("#### Categorical distribution comparison")
            selected_syn_cat = st.selectbox("Categorical variable", shared_cat, key="syn_cat")
            cat_comp=pd.concat([raw[selected_syn_cat].value_counts(normalize=True,dropna=False).rename("Original"),syn[selected_syn_cat].value_counts(normalize=True,dropna=False).rename("Synthetic")],axis=1).fillna(0)
            st.dataframe((cat_comp*100).round(2),use_container_width=True)
            fig,ax=plt.subplots(figsize=(8,4)); (cat_comp*100).plot(kind="bar",ax=ax); ax.set_ylabel("Share (%)"); ax.set_title(f"Original vs synthetic — {selected_syn_cat}"); ax.tick_params(axis="x",rotation=35); st.pyplot(fig); plt.close(fig)
        st.download_button("Download synthetic data CSV",syn.to_csv(index=False).encode(),"Dean_Dilemma_Synthetic.csv","text/csv")

# Tab 7 risk scoring
with tabs[5]:
    st.subheader("Individual student risk scoring")
    bundle=st.session_state.get("model_bundle")
    if not bundle:
        st.info("Train models first in the Classification tab.")
    else:
        model_name=st.selectbox("Model for scoring",list(bundle["models"].keys()),key="risk_model")
        model=bundle["models"][model_name]
        with st.form("student_form"):
            st.write("Enter a student's predictor values. Missing values can be left blank where supported.")
            vals={}
            cols=st.columns(2)
            for i,c in enumerate(feature_cols):
                with cols[i%2]:
                    if c in num_cols:
                        median=float(raw[c].median()) if raw[c].notna().any() else 0.0
                        vals[c]=st.number_input(c,value=median,key=f"risk_{c}")
                    else:
                        choices=sorted(raw[c].dropna().astype(str).unique().tolist())
                        vals[c]=st.selectbox(c,choices,key=f"risk_{c}") if choices else "Unknown"
            submitted=st.form_submit_button("Predict placement probability",type="primary")
        if submitted:
            row=pd.DataFrame([vals],columns=feature_cols); prob=float(model.predict_proba(row)[0,1]); predicted=int(prob>=float(threshold))
            left,right=st.columns(2); left.metric("Estimated probability of placement",f"{prob:.1%}"); right.metric("Predicted category",target_labels[predicted])
            st.progress(min(max(prob,0.0),1.0))
            if predicted==0: st.warning("The model flags this profile as potentially at risk. This is a screening signal, not a final judgement about the student.")
            else: st.success("The model predicts placement for this profile. Continue normal career support and monitoring.")
            st.caption("This score is a model estimate based on historical associations; it is not causal and should not be used as the sole basis for decisions.")

# Tab 8 downloads/method
with tabs[6]:
    st.subheader("Export and methodology")
    st.markdown("**Workflow reproduced from the cleaned notebook:**")
    st.markdown("1. Load workbook and inspect data types, target distribution and missing values.\n2. CCA diagnostic quantifies records that would be lost by complete-case deletion.\n3. Outlier diagnostics: IQR, Z-score, percentile and Isolation Forest.\n4. Stratified train/test split; predictors exclude identifiers, placement text and salary.\n5. Preprocessing: median or MICE for numerical variables; mode for categorical variables; scaling and one-hot encoding.\n6. Classification: Logistic Regression, LDA, Random Forest and AdaBoost; optional 5-fold CV and RF tuning.\n7. Model evaluation: accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, MCC, confusion matrix, ROC and PR curves.\n8. Optional SDV Gaussian Copula synthetic-data generation and distribution comparisons.")
    st.markdown("**Important limitations**")
    st.markdown("- 391 original records are a small sample; validation uncertainty can be high.\n- A high accuracy can hide failure on the minority class.\n- Synthetic records are not additional real students.\n- If synthetic data is generated from the full original dataset, evaluating against a holdout from that same source can introduce leakage.\n- Risk scores should support, not replace, human review.")
    st.download_button("Download cleaned analysis dataset",raw.to_csv(index=False).encode(),"Dean_Dilemma_Cleaned.csv","text/csv")
    st.download_button("Download missing-value report",pd.DataFrame({"Variable":feature_cols,"Missing":raw[feature_cols].isna().sum().values,"Missing %":(raw[feature_cols].isna().mean()*100).values}).to_csv(index=False).encode(),"missing_value_report.csv","text/csv")
    if "model_bundle" in st.session_state:
        st.download_button("Download model metrics",st.session_state["model_bundle"]["results"].to_csv(index=False).encode(),"model_metrics.csv","text/csv")

st.divider()
st.caption("Dean's Dilemma Analytics Lab · Local prototype · Results are calculated from the workbook currently loaded in this session.")
