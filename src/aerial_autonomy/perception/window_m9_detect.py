"""Classical color-mask and four-line extraction from rendered RGB pixels only."""

from dataclasses import dataclass

import numpy as np

from aerial_autonomy.perception.camera import CameraCalibration


@dataclass(frozen=True)
class WindowDetection:
    window_id: str
    valid: bool
    reason: str | None
    corners_uv: np.ndarray | None
    confidence: float
    color_pixel_count: int
    line_rmse_px: float | None


def _runs(indices: np.ndarray):
    if len(indices)==0:
        return []
    cuts=np.flatnonzero(np.diff(indices)>1)+1
    return np.split(indices,cuts)


def _fit_line(x: np.ndarray,y: np.ndarray):
    if len(x)<12 or np.ptp(x)<10:
        return None
    used=np.ones(len(x),dtype=bool)
    for _ in range(3):
        coefficients=np.polyfit(x[used],y[used],1)
        residual=abs(y-(coefficients[0]*x+coefficients[1]))
        keep=residual<max(2.,3*np.median(residual[used]))
        if np.array_equal(keep,used):
            break
        used=keep
        if np.count_nonzero(used)<12:
            return None
    rmse=float(np.sqrt(np.mean((y[used]-np.polyval(coefficients,x[used]))**2)))
    return coefficients,rmse


def _intersection(vertical,horizontal):
    a,b=vertical
    c,d=horizontal
    if abs(1-a*c)<1e-5:
        return None
    u=(a*d+b)/(1-a*c)
    return np.array([u,c*u+d])


class ColorWindowDetector:
    """Consumes only pixels, target ID, known appearance, and expected view side."""

    def __init__(self,colors_by_id: dict[str,tuple[int,int,int]]):
        self.colors_by_id=dict(colors_by_id)
        dominant={key:int(np.argmax(color)) for key,color in colors_by_id.items()}
        if len(set(dominant.values()))!=len(dominant):
            raise ValueError("window colors need distinct dominant RGB channels")

    def detect(self,image_rgb: np.ndarray,camera: CameraCalibration,window_id: str,
               expected_side: str="positive_normal") -> WindowDetection:
        if window_id not in self.colors_by_id:
            return WindowDetection(window_id,False,"unknown_window_id",None,0,0,None)
        if expected_side not in ("positive_normal","negative_normal"):
            return WindowDetection(window_id,False,"invalid_side",None,0,0,None)
        image=np.asarray(image_rgb)
        if (image.shape!=(camera.height_px,camera.width_px,3)
                or image.dtype!=np.uint8):
            return WindowDetection(window_id,False,"invalid_image",None,0,0,None)
        channel=int(np.argmax(self.colors_by_id[window_id]))
        competitors=[i for i in range(3) if i!=channel]
        values=image.astype(np.int16)
        mask=(values[:,:,channel]>115)
        for other in competitors:
            mask &= values[:,:,channel]>1.55*values[:,:,other]
        # Three-by-three majority removes isolated noisy pixels without any
        # 3D geometry, oracle projections, or external vision dependency.
        padded=np.pad(mask.astype(np.uint8),1)
        neighbors=sum(padded[dy:dy+mask.shape[0],dx:dx+mask.shape[1]]
                      for dy in range(3) for dx in range(3))
        mask=neighbors>=5
        count=int(np.count_nonzero(mask))
        def fail(reason):
            return WindowDetection(window_id,False,reason,None,0.,count,None)
        if count<100:
            return fail("no_frame_pixels")
        left=[];right=[];row_y=[]
        for v in range(mask.shape[0]):
            runs=[r for r in _runs(np.flatnonzero(mask[v])) if len(r)>=2]
            if len(runs)>=2 and runs[-1][0]-runs[0][-1]>=15:
                row_y.append(v+.5)
                left.append(runs[0][-1]+.5)
                right.append(runs[-1][0]+.5)
        top=[];bottom=[];col_x=[]
        for u in range(mask.shape[1]):
            runs=[r for r in _runs(np.flatnonzero(mask[:,u])) if len(r)>=2]
            if len(runs)>=2 and runs[-1][0]-runs[0][-1]>=15:
                col_x.append(u+.5)
                top.append(runs[0][-1]+.5)
                bottom.append(runs[-1][0]+.5)
        if min(len(row_y),len(col_x))<12:
            return fail("incomplete_frame")
        line_fits=(_fit_line(np.asarray(row_y),np.asarray(left)),
                   _fit_line(np.asarray(row_y),np.asarray(right)),
                   _fit_line(np.asarray(col_x),np.asarray(top)),
                   _fit_line(np.asarray(col_x),np.asarray(bottom)))
        if any(x is None for x in line_fits):
            return fail("line_fit_failed")
        (left_fit,left_err),(right_fit,right_err),(top_fit,top_err),(bottom_fit,bottom_err)=line_fits
        tl=_intersection(left_fit,top_fit)
        tr=_intersection(right_fit,top_fit)
        bl=_intersection(left_fit,bottom_fit)
        br=_intersection(right_fit,bottom_fit)
        if any(x is None or not np.all(np.isfinite(x)) for x in (tl,tr,bl,br)):
            return fail("degenerate_quadrilateral")
        corners=np.array([bl,br,tr,tl])
        if expected_side=="negative_normal":
            corners=np.array([br,bl,tl,tr])
        width=float((np.linalg.norm(tr-tl)+np.linalg.norm(br-bl))/2)
        height=float((np.linalg.norm(tl-bl)+np.linalg.norm(tr-br))/2)
        if (width<18 or height<18 or np.any(corners[:,0]<1) or
                np.any(corners[:,0]>=camera.width_px-1) or np.any(corners[:,1]<1)
                or np.any(corners[:,1]>=camera.height_px-1)):
            return fail("clipped_or_small_opening")
        center=np.mean(corners,axis=0)
        if mask[int(center[1]),int(center[0])]:
            return fail("solid_color_region")
        line_rmse=float(np.mean([left_err,right_err,top_err,bottom_err]))
        if line_rmse>2.5:
            return fail("poor_line_consistency")
        confidence=float(np.clip(1-line_rmse/3,0,1)*np.clip(min(width,height)/50,0,1))
        return WindowDetection(window_id,True,None,corners,confidence,count,line_rmse)
