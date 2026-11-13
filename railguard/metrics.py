"""Binary classification and object-box matching metrics."""
import numpy as np


FAILURE_CATEGORIES = (
    "occlusion", "small_person", "motion_blur", "near_train_door",
    "missing_feet", "illumination_extreme", "camera_tilt",
)


def validate_failure_categories(categories) -> tuple[str, ...]:
    """Validate and deduplicate the fixed P3 failure taxonomy."""
    result = tuple(dict.fromkeys(str(category) for category in categories))
    unknown = sorted(set(result) - set(FAILURE_CATEGORIES))
    if unknown:
        raise ValueError(f"Unknown failure categories: {', '.join(unknown)}")
    return result


def binary_metrics(truth, predicted) -> dict:
    a=np.asarray(truth,dtype=int)
    b=np.asarray(predicted,dtype=int)
    if len(a)!=len(b) or len(a)==0:
        raise ValueError("Expected equal-length nonempty labels")
    tp=int(((a==1)&(b==1)).sum())
    tn=int(((a==0)&(b==0)).sum())
    fp=int(((a==0)&(b==1)).sum())
    fn=int(((a==1)&(b==0)).sum())
    div=lambda x,y: float(x/y) if y else 0.0
    prec=div(tp,tp+fp)
    rec=div(tp,tp+fn)
    f1=div(2*prec*rec,prec+rec)
    return {"n":len(a),"tp":tp,"tn":tn,"fp":fp,"fn":fn,"accuracy":div(tp+tn,len(a)),
            "precision":prec,"recall":rec,"f1":f1,"specificity":div(tn,tn+fp)}


def bbox_iou(a,b) -> float:
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    intersection=max(0.,x2-x1)*max(0.,y2-y1)
    aa=max(0.,a[2]-a[0])*max(0.,a[3]-a[1])
    bb=max(0.,b[2]-b[0])*max(0.,b[3]-b[1])
    return intersection/max(aa+bb-intersection,1e-9)


def greedily_match(gt, pred, threshold=.5):
    """One-to-one max IoU matching, returns [(gt_idx,pred_idx,iou)], missed_gt, excess_pred."""
    pairs=sorted(((bbox_iou(g.bbox,p.bbox),i,j) for i,g in enumerate(gt)
                  for j,p in enumerate(pred)),reverse=True)
    matched=[]; gi=set(); pj=set()
    for iou,i,j in pairs:
        if iou < threshold:
            break
        if i not in gi and j not in pj:
            matched.append((i,j,iou));gi.add(i);pj.add(j)
    return matched,[i for i in range(len(gt)) if i not in gi],[j for j in range(len(pred)) if j not in pj]
