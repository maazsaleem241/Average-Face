import os
import cv2
import dlib
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from diffusers import StableDiffusionXLImg2ImgPipeline
from ultralytics import YOLO

# ============================================================
# CONSTANTS & DIRECTORY CONFIGURATION
# ============================================================
BASE_DIR = r"D:\Intern work\Data for research\data_rearranged"

GROUP_MAPPING = {
    'Group_A': ['pr_8', 'pr_17', 'pr_15', 'pr_4', 'pr_1', 'pr_22'],
    'Group_B': ['pr_9', 'pr_23', 'pr_3', 'pr_10', 'pr_21', 'pr_11']
}

EXPRESSIONS = [
    'Attentional_Engagement',
    'Aversion',
    'Concentration',
    'Dejection',
    'Positive_Social_Expression',
    'Skepticism',
    'Startle_Response',
    'Tension_Stress'
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

# ============================================================
# CORE ARCHITECTURE DETECTORS
# ============================================================
print("-> Loading YOLO Architecture Engine...")
yolo_model_path = os.path.join(SCRIPT_DIR, "yolov8n-face.pt")
if not os.path.exists(yolo_model_path):
    raise FileNotFoundError(f"Missing YOLO weights! Place in: {SCRIPT_DIR}")
yolo_model = YOLO(yolo_model_path)

print("-> Loading Dlib CNN Face Detector & Predictor...")
cnn_model_path = os.path.join(SCRIPT_DIR, "mmod_human_face_detector.dat")
if not os.path.exists(cnn_model_path):
    raise FileNotFoundError(f"Missing CNN weights! Place mmod_human_face_detector.dat in: {SCRIPT_DIR}")
detector = dlib.cnn_face_detection_model_v1(cnn_model_path)

predictor_path = os.path.join(SCRIPT_DIR, "shape_predictor_68_face_landmarks.dat")
if not os.path.exists(predictor_path):
    raise FileNotFoundError(f"Missing weights! Place shape_predictor_68_face_landmarks.dat in: {SCRIPT_DIR}")
predictor = dlib.shape_predictor(predictor_path)


# ============================================================
# MATHEMATICAL WARPING & ALIGNMENT GEOMETRY
# ============================================================
def get_quadratic_bezier_points(p0, p1, p2, num_points=20):
    t = np.linspace(0, 1, num_points)[:, None]
    return (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2

def align_face_by_eyes(image, landmarks):
    left_eye = np.mean(landmarks[36:42], axis=0)
    right_eye = np.mean(landmarks[42:48], axis=0)
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    current_dist = np.sqrt(dx**2 + dy**2)
    desired_dist = 220
    scale = desired_dist / current_dist
    eyes_center = (float((left_eye[0] + right_eye[0]) / 2), float((left_eye[1] + right_eye[1]) / 2))
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    M[0, 2] += 300 - eyes_center[0]
    M[1, 2] += 250 - eyes_center[1]
    aligned = cv2.warpAffine(image, M, (600, 600), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    landmarks_h = np.hstack([landmarks, np.ones((68, 1))])
    transformed = (M @ landmarks_h.T).T
    return aligned, transformed

def normalize_landmarks(landmarks):
    centered = landmarks - np.mean(landmarks, axis=0)
    scale = np.sqrt(np.sum(centered**2))
    return centered / scale if scale != 0 else centered

def warp_triangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    t1_rect = [((t1[i][0] - r1[0]), (t1[i][1] - r1[1])) for i in range(3)]
    t2_rect = [((t2[i][0] - r2[0]), (t2[i][1] - r2[1])) for i in range(3)]
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    if img1_rect.size == 0: return
    warp_mat = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
    img2_rect = cv2.warpAffine(img1_rect, warp_mat, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0), 16, 0)
    target_slice = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    th, tw = target_slice.shape[:2]
    if img2_rect.shape[0] != th or img2_rect.shape[1] != tw:
        img2_rect = cv2.resize(img2_rect, (tw, th), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = target_slice * (1 - mask) + img2_rect * mask

def warp_image(img, landmarks, target_landmarks, size):
    output_img = img.copy()
    rect = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(rect)
    for p in target_landmarks:
        subdiv.insert((int(p[0]), int(p[1])))
    tri_list = subdiv.getTriangleList()
    for t in tri_list:
        pt = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        idx = []
        is_valid_triangle = True
        for p in pt:
            dists = np.linalg.norm(target_landmarks - p, axis=1)
            min_idx = np.argmin(dists)
            if dists[min_idx] < 2.0:
                idx.append(min_idx)
            else:
                is_valid_triangle = False
                break
        if is_valid_triangle and len(idx) == 3:
            t1 = [landmarks[idx[0]], landmarks[idx[1]], landmarks[idx[2]]]
            t2 = [target_landmarks[idx[0]], target_landmarks[idx[1]], target_landmarks[idx[2]]]
            warp_triangle(img, output_img, t1, t2)
    return output_img

def pyramid_blend(warped_images):
    levels = 3 
    num_images = len(warped_images)
    gaussian_pyramids = []
    for img in warped_images:
        G = img.copy()
        gp = [G]
        for j in range(levels):
            G = cv2.pyrDown(G)
            gp.append(G)
        gaussian_pyramids.append(gp)
    laplacian_pyramids = []
    for i in range(num_images):
        gp = gaussian_pyramids[i]
        lp = [gp[levels-1]]
        for j in range(levels-1, 0, -1):
            GE = cv2.pyrUp(gp[j])
            GE = cv2.resize(GE, (gp[j-1].shape[1], gp[j-1].shape[0]))
            L = cv2.subtract(gp[j-1], GE)
            lp.append(L)
        laplacian_pyramids.append(lp)
    blended_pyramid = []
    for level_idx in range(levels):
        level_layers = [laplacian_pyramids[img_idx][level_idx] for img_idx in range(num_images)]
        avg_level_layer = np.mean(level_layers, axis=0)
        blended_pyramid.append(avg_level_layer)
    ls_ = blended_pyramid[0]
    for i in range(1, levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.resize(ls_, (blended_pyramid[i].shape[1], blended_pyramid[i].shape[0]))
        ls_ = cv2.add(ls_, blended_pyramid[i])
    return np.clip(ls_, 0.0, 1.0)


# ============================================================
# PROCESSING PIPELINE PIPES
# ============================================================
def run_morphic_merger(selected_img_paths, target_group, current_expression):
    print(f"\n --> Calculating average face for {target_group} - {current_expression}...")
    
    output_base_path = os.path.join(BASE_DIR, "Groups", target_group, "average_face", "outputs", current_expression)
    yolo_crop_path = os.path.join(output_base_path, "yolo_detections")
    overlay_path = os.path.join(output_base_path, "facedetector_overlay")
    
    for p in [output_base_path, yolo_crop_path, overlay_path]:
        os.makedirs(p, exist_ok=True)
        
    output_size = (600, 600, 3)
    images = []
    all_landmarks = []
    
    for img_path in selected_img_paths:
        file = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None: continue
        
        results = yolo_model(img, verbose=False)
        boxes = results[0].boxes
        if len(boxes) == 0: continue
            
        best_box = max(boxes, key=lambda b: b.conf[0].item())
        xyxy = best_box.xyxy[0].cpu().numpy().astype(int)
        img_h, img_w = img.shape[:2]
        
        padding = 0.105
        w = xyxy[2] - xyxy[0]
        h = xyxy[3] - xyxy[1]
        box_x1 = max(0, int(xyxy[0] - w * padding))
        box_y1 = max(0, int(xyxy[1] - h * padding))
        box_x2 = min(img_w, int(xyxy[2] + w * padding))
        box_y2 = min(img_h, int(xyxy[3] + h * padding))
        
        yolo_cropped = img[box_y1:box_y2, box_x1:box_x2]
        yolo_cropped = cv2.resize(yolo_cropped, (600, 600))
        cv2.imwrite(os.path.join(yolo_crop_path, f"yolo_{file}"), yolo_cropped)
        
        gray_crop = cv2.cvtColor(yolo_cropped, cv2.COLOR_BGR2GRAY)
        detections = detector(gray_crop, 0)
        if len(detections) == 0: continue
            
        best_detection = max(detections, key=lambda d: d.confidence)
        dlib_rect = best_detection.rect
        shape = predictor(gray_crop, dlib_rect)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        
        aligned_img, aligned_landmarks = align_face_by_eyes(yolo_cropped, landmarks)
        
        overlay_img = aligned_img.copy()
        for n in range(0, 68):
            cv2.circle(overlay_img, (int(aligned_landmarks[n, 0]), int(aligned_landmarks[n, 1])), 3, (0, 255, 0), -1)
        cv2.imwrite(os.path.join(overlay_path, f"detected_{file}"), overlay_img)
        
        images.append(aligned_img.astype(np.float32) / 255.0)
        all_landmarks.append(aligned_landmarks)
        
    if len(images) < 2:
        return None, None

    normalized_landmarks = [normalize_landmarks(lm) for lm in all_landmarks]
    mean_shape = np.mean(normalized_landmarks, axis=0)
    min_x, min_y = np.min(mean_shape, axis=0)
    max_x, max_y = np.max(mean_shape, axis=0)
    
    scale_factor = 410.0 / (max_y - min_y)
    scaled_landmarks = (mean_shape - [min_x, min_y]) * scale_factor
    left_eye_avg = np.mean(scaled_landmarks[36:42], axis=0)
    right_eye_avg = np.mean(scaled_landmarks[42:48], axis=0)
    eyes_center_avg = (left_eye_avg + right_eye_avg) / 2.0
    center_offset = np.array([300.0, 250.0]) - eyes_center_avg
    avg_landmarks = scaled_landmarks + center_offset
    
    h_600, w_600 = 600, 600
    corners = np.array([[0, 0], [w_600 // 2, 0], [w_600 - 1, 0], [0, h_600 // 2], [w_600 - 1, h_600 // 2], [0, h_600 - 1], [w_600 // 2, h_600 - 1], [w_600 - 1, h_600 - 1]])
    
    avg_landmarks_extended = np.vstack([avg_landmarks, corners])
    all_landmarks_extended = [np.vstack([lm, corners]) for lm in all_landmarks]
        
    processed_warped_images = []
    for i in range(len(images)):
        warped_img = warp_image(images[i], all_landmarks_extended[i], avg_landmarks_extended, output_size)
        processed_warped_images.append(warped_img)
        
    blended_face = pyramid_blend(processed_warped_images)
    blended_uint8 = (blended_face * 255).astype(np.uint8)
    avg_warped_base = np.mean(processed_warped_images, axis=0)
    avg_warped_uint8 = (avg_warped_base * 255).astype(np.uint8)
    
    sharp_feature_mask = np.zeros((600, 600), dtype=np.uint8)
    left_eye_poly = avg_landmarks[36:42].astype(np.int32)
    right_eye_poly = avg_landmarks[42:48].astype(np.int32)
    outer_lip_poly = avg_landmarks[48:60].astype(np.int32)
    cv2.fillConvexPoly(sharp_feature_mask, left_eye_poly, 255)
    cv2.fillConvexPoly(sharp_feature_mask, right_eye_poly, 255)
    cv2.fillConvexPoly(sharp_feature_mask, outer_lip_poly, 255)
    sharp_feature_mask = cv2.GaussianBlur(sharp_feature_mask, (15, 15), 0)
    feature_mask_normalized = (sharp_feature_mask.astype(np.float32) / 255.0)[:, :, None]
    
    normalized_face_img = (blended_uint8 * (1.0 - feature_mask_normalized) + avg_warped_uint8 * feature_mask_normalized)
    final_combined_face = np.clip(normalized_face_img, 0, 255).astype(np.uint8)
    pure_unrefined_face = final_combined_face.copy()
    
    brow_avg_y = np.mean(avg_landmarks[17:27], axis=0)[1]
    eye_avg_y = np.mean(avg_landmarks[36:48], axis=0)[1]
    forehead_height = (eye_avg_y - brow_avg_y) * 2.2
    left_temple = avg_landmarks[0]
    right_temple = avg_landmarks[16]
    mid_brow = avg_landmarks[21]
    forehead_left_ctrl = np.array([left_temple[0], left_temple[1] - forehead_height * 0.75])
    forehead_apex = np.array([mid_brow[0], mid_brow[1] - forehead_height])
    forehead_right_ctrl = np.array([right_temple[0], right_temple[1] - forehead_height * 0.75])
    left_curve = get_quadratic_bezier_points(left_temple, forehead_left_ctrl, forehead_apex, num_points=15)
    right_curve = get_quadratic_bezier_points(forehead_apex, forehead_right_ctrl, right_temple, num_points=15)
    forehead_spline = np.vstack([left_curve, right_curve[1:]])
    face_profile = np.vstack([avg_landmarks[0:17], forehead_spline[::-1]])
    
    clean_expr_name = current_expression.lower()
    unrefined_filename = f"{target_group.lower()}_{clean_expr_name}_pre_sdxl_average_face.jpg"
    refined_filename = f"{target_group.lower()}_{clean_expr_name}_post_sdxl_average_face.jpg"
    
    unrefined_output_path = os.path.join(output_base_path, unrefined_filename)
    cv2.imwrite(unrefined_output_path, final_combined_face)
    
    print("\nBooting Stable Diffusion XL Local Architecture Passing Loops...")
    try:
        model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=True).to("cpu")
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
        
        init_image = Image.open(unrefined_output_path).convert("RGB").resize((512, 512))
        prompt = "A professional, ultra-high-resolution passport portrait photograph of a real human face, sharp focus, explicit detailed skin pores and subtle textures, completely realistic human eyes, uniform studio background"
        negative_prompt = "glasses, spectacles, waxy skin, smoothed face, graphic art, 3D render, cartoon, digital illustration, ghosting artifacts, extra lines around eyes"
        
        refined_image = pipe(prompt=prompt, negative_prompt=negative_prompt, image=init_image, strength=0.12, guidance_scale=8.0).images[0]
        refined_cv2 = cv2.cvtColor(np.array(refined_image), cv2.COLOR_RGB2BGR)
        refined_cv2 = cv2.resize(refined_cv2, (600, 600))
        
        face_hull_mask = np.zeros((600, 600), dtype=np.uint8)
        cv2.fillConvexPoly(face_hull_mask, face_profile.astype(np.int32), 255)
        face_hull_mask = cv2.GaussianBlur(face_hull_mask, (11, 11), 0)
        face_mask_normalized = (face_hull_mask.astype(np.float32) / 255.0)[:, :, None]
        refined_cv2 = (refined_cv2 * face_mask_normalized + pure_unrefined_face * (1.0 - face_mask_normalized)).astype(np.uint8)
        
        lip_mask = np.zeros((600, 600), dtype=np.uint8)
        cv2.fillConvexPoly(lip_mask, avg_landmarks[48:60].astype(np.int32), 255)
        lip_mask = cv2.GaussianBlur(lip_mask, (11, 11), 0)
        lip_mask_normalized = (lip_mask.astype(np.float32) / 255.0)[:, :, None]
        final_grafted_face = (refined_cv2 * (1.0 - lip_mask_normalized) + pure_unrefined_face * lip_mask_normalized).astype(np.uint8)
        
        refined_output_path = os.path.join(output_base_path, refined_filename)
        cv2.imwrite(refined_output_path, final_grafted_face)
        return unrefined_output_path, refined_output_path
    except Exception as e:
        print(f"⚠️ SDXL Engine Offline: {e}")
        return unrefined_output_path, None


# ============================================================
# INTERACTIVE PRODUCTION DISPLAY RUNTIME
# ============================================================
class ProductionPipelineGUI:
    def __init__(self, root, target_group, starting_expression):
        self.root = root
        self.target_group = target_group
        self.participants = GROUP_MAPPING[target_group]
        
        self.root.title(f"Average Face Pipeline - {self.target_group}")
        self.root.geometry("1100x820") 
        self.root.configure(bg="#2c3e50")
        
        if starting_expression in EXPRESSIONS:
            self.expr_idx = EXPRESSIONS.index(starting_expression)
        else:
            self.expr_idx = 0
            
        self.p_idx = 0
        self.f_index = 0
        self.in_review_mode = False  
        
        self.current_p_frames = []
        self.selected_manifest = [] 
        self.current_unrefined_file = None
        self.current_refined_file = None
        
        self.setup_ui()
        self.bind_keys()
        self.load_active_expression_sequence()

    def setup_ui(self):
        self.meta_frame = tk.Frame(self.root, bg="#34495e", height=65)
        self.meta_frame.pack(fill=tk.X)
        
        self.lbl_expr_info = tk.Label(self.meta_frame, text="Expression: --/--", font=("Arial", 11, "bold"), fg="#e74c3c", bg="#34495e")
        self.lbl_expr_info.pack(anchor=tk.W, padx=20, pady=5)
        
        self.lbl_p_info = tk.Label(self.meta_frame, text="Participant Index: --/--", font=("Arial", 11), fg="#ecf0f1", bg="#34495e")
        self.lbl_p_info.pack(side=tk.LEFT, padx=20, pady=5)
        
        self.lbl_f_info = tk.Label(self.meta_frame, text="Reviewing Frame File: --/--", font=("Arial", 10), fg="#bdc3c7", bg="#34495e")
        self.lbl_f_info.pack(side=tk.RIGHT, padx=20, pady=5)
        
        self.canvas_frame = tk.Frame(self.root, bg="#1a252f")
        self.canvas_frame.pack(expand=True, fill=tk.BOTH, padx=30, pady=20)
        
        self.help_frame = tk.Frame(self.root, bg="#2c3e50")
        self.help_frame.pack(fill=tk.X, pady=10)
        self.lbl_help = tk.Label(self.help_frame, text="", font=("Arial", 11, "italic"), fg="#ecf0f1", bg="#2c3e50")
        self.lbl_help.pack()

    def bind_keys(self):
        self.root.bind("<Left>", lambda e: self.move_left())
        self.root.bind("<Right>", lambda e: self.move_right())
        self.root.bind("<Return>", lambda e: self.handle_enter())
        self.root.bind("<space>", lambda e: self.handle_enter())  
        self.root.bind("<BackSpace>", lambda e: self.handle_backspace())
        self.root.bind("<q>", lambda e: self.safe_quit_trigger())
        self.root.bind("<Q>", lambda e: self.safe_quit_trigger())

    def update_help_instructions(self):
        if self.in_review_mode:
            text = "📋 EXPR COMPLETE: Looks correct?  |  <- Enter / Space: Yes (Advance)  |  X Backspace: No (Wipe & Redo)  |  [Q]: Quit"
            self.lbl_help.config(text=text, fg="#f1c40f")
        else:
            text = "Controls:  <- Arrow Left (Prev)  |  -> Arrow Right (Next)  |  <- Enter (Select Frame)  |  [Q]: Exit Pipeline"
            self.lbl_help.config(text=text, fg="#ecf0f1")

    def load_active_expression_sequence(self):
        if self.expr_idx >= len(EXPRESSIONS):
            messagebox.showinfo("Campaign Finished", f"All profiles for {self.target_group} are permanently compiled!")
            self.root.destroy()
            return
            
        self.current_expression = EXPRESSIONS[self.expr_idx]
        self.p_idx = 0
        self.selected_manifest = []
        self.in_review_mode = False
        
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
            
        self.img_label_left = tk.Label(self.canvas_frame, bg="#1a252f")
        self.img_label_left.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        self.update_help_instructions()
        self.load_participant_folder()

    def load_participant_folder(self):
        if self.p_idx >= len(self.participants):
            self.trigger_morphing_sequence()
            return
            
        current_p_id = self.participants[self.p_idx]
        
        # SCRIPT CORRECTION: Check your exact localized directory path map from the snapshot
        group_data_dir = os.path.join(BASE_DIR, "Groups", self.target_group, "data")
        evidence_csv_path = os.path.join(group_data_dir, f"{current_p_id}_event_evidence.csv")
        
        # Verify if evidence configuration sheet exists
        if not os.path.exists(evidence_csv_path):
            print(f"⚠️ Missing event evidence map at: {evidence_csv_path}, skipping participant...")
            self.p_idx += 1
            self.load_participant_folder()
            return
            
        # fallback path fallback execution structure to look at your framework frames
        self.current_p_dir = os.path.join(BASE_DIR, "Frequency Analysis", current_p_id, "frames", self.current_expression)
        
        if os.path.exists(self.current_p_dir):
            self.current_p_frames = sorted([f for f in os.listdir(self.current_p_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        else:
            self.current_p_frames = []
            
        if not self.current_p_frames:
            print(f"⚠️ Frame buffer empty for {current_p_id} under {self.current_expression}")
            self.p_idx += 1
            self.load_participant_folder()
            return
            
        self.f_index = 0
        self.display_current_frame()

    def display_current_frame(self):
        current_p_id = self.participants[self.p_idx]
        filename = self.current_p_frames[self.f_index]
        full_img_path = os.path.join(self.current_p_dir, filename)
        
        self.lbl_expr_info.config(text=f"Active Global Sweeper Phase: {self.current_expression.upper()} ({self.expr_idx + 1}/{len(EXPRESSIONS)})")
        self.lbl_p_info.config(text=f"Participant Index: {current_p_id} ({self.p_idx + 1}/{len(self.participants)})")
        self.lbl_f_info.config(text=f"Reviewing Frame: {self.f_index + 1} / {len(self.current_p_frames)}")
        
        try:
            img = Image.open(full_img_path)
            img.thumbnail((750, 480))
            self.tk_img_left = ImageTk.PhotoImage(img)
            self.img_label_left.config(image=self.tk_img_left, text="")
        except Exception:
            pass

    def move_right(self):
        if self.in_review_mode or not self.current_p_frames: return
        self.f_index = (self.f_index + 1) % len(self.current_p_frames)
        self.display_current_frame()

    def move_left(self):
        if self.in_review_mode or not self.current_p_frames: return
        self.f_index = (self.f_index - 1 + len(self.current_p_frames)) % len(self.current_p_frames)
        self.display_current_frame()

    def handle_enter(self):
        if self.in_review_mode:
            print(f"Average Face Accepted: Moving to next expression segment pass.")
            self.expr_idx += 1
            self.load_active_expression_sequence()
        else:
            current_p_id = self.participants[self.p_idx]
            chosen_filename = self.current_p_frames[self.f_index]
            full_target_path = os.path.join(self.current_p_dir, chosen_filename)
            
            self.selected_manifest.append(full_target_path)
            print(f"    💾 SELECTION RECORDED: Loaded {current_p_id} ➔ {chosen_filename}")
            
            self.p_idx += 1
            self.load_participant_folder()

    def handle_backspace(self):
        if not self.in_review_mode: return
        print(f"❌ ARCHETYPE REJECTED: Purging expression calculation buffer for: {self.current_expression}")
        
        target_dir = os.path.join(BASE_DIR, "Groups", self.target_group, "average_face", "outputs", self.current_expression)
        if os.path.exists(target_dir):
            import shutil
            try: shutil.rmtree(target_dir)
            except Exception as e: print(f"Purge note: {e}")
                
        self.load_active_expression_sequence()

    def trigger_morphing_sequence(self):
        messagebox.showinfo("Processing Array Matrix", f"All frames for {self.current_expression} have been selected. Click okay to start the facial morphing pipeline. This may take a few moments depending on the number of participants and system performance.")
        unrefined, refined = run_morphic_merger(self.selected_manifest, self.target_group, self.current_expression)
        
        self.current_unrefined_file = unrefined
        self.current_refined_file = refined
        self.render_review_screen()

    def render_review_screen(self):
        self.in_review_mode = True
        self.update_help_instructions()
        
        self.lbl_p_info.config(text=f"Evaluation Window: {self.current_expression.upper()} - {self.target_group}")
        self.lbl_f_info.config(text="Status Check: Target Outputs Compiled")
        
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
            
        left_subframe = tk.Frame(self.canvas_frame, bg="#1a252f")
        left_subframe.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10)
        
        lbl_left_title = tk.Label(left_subframe, text="Before Refinement", font=("Arial", 13, "bold"), fg="#3498db", bg="#1a252f")
        lbl_left_title.pack(side=tk.TOP, pady=(0, 10))
        
        self.img_label_left = tk.Label(left_subframe, bg="#1a252f")
        self.img_label_left.pack(side=tk.TOP, expand=True, fill=tk.BOTH)
        
        right_subframe = tk.Frame(self.canvas_frame, bg="#1a252f")
        right_subframe.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10)
        
        lbl_right_title = tk.Label(right_subframe, text="After Refinement", font=("Arial", 13, "bold"), fg="#2ecc71", bg="#1a252f")
        lbl_right_title.pack(side=tk.TOP, pady=(0, 10))
        
        self.img_label_right = tk.Label(right_subframe, bg="#1a252f")
        self.img_label_right.pack(side=tk.TOP, expand=True, fill=tk.BOTH)
        
        if self.current_unrefined_file and os.path.exists(self.current_unrefined_file):
            try:
                img_l = Image.open(self.current_unrefined_file)
                img_l.thumbnail((480, 480))
                self.tk_review_l = ImageTk.PhotoImage(img_l)
                self.img_label_left.config(image=self.tk_review_l)
            except Exception:
                pass
                
        target_refined = self.current_refined_file if self.current_refined_file else self.current_unrefined_file
        if target_refined and os.path.exists(target_refined):
            try:
                img_r = Image.open(target_refined)
                img_r.thumbnail((480, 480))
                self.tk_review_r = ImageTk.PhotoImage(img_r)
                self.img_label_right.config(image=self.tk_review_r)
            except Exception:
                pass

    def safe_quit_trigger(self):
        ans = messagebox.askyesno("Exit Operational Pipeline", "Are you sure you want to quit the filtering interface?")
        if ans:
            self.root.quit()
            self.root.destroy()


# ============================================================
# INITIAL ENTRY POINT SELECTOR DIALOG
# ============================================================
class CohortSelectorWindow:
    def __init__(self, master):
        self.master = master
        self.master.title("Group Selector Panel")
        self.master.geometry("450x320") 
        self.master.configure(bg="#34495e")
        
        lbl_welcome = tk.Label(master, text="Average Face Pipeline Launcher", font=("Arial", 12, "bold"), fg="#ecf0f1", bg="#34495e")
        lbl_welcome.pack(pady=(20, 10))
        
        lbl_prompt = tk.Label(master, text="Select your group:", font=("Arial", 11), fg="#bdc3c7", bg="#34495e")
        lbl_prompt.pack(pady=5)
        
        self.group_var = tk.StringVar()
        self.combo_group = ttk.Combobox(master, textvariable=self.group_var, values=list(GROUP_MAPPING.keys()), state="readonly", font=("Arial", 11), width=25)
        self.combo_group.pack(pady=5)
        self.combo_group.current(0)
        
        lbl_expr_prompt = tk.Label(master, text="Select Starting Expression Phase:", font=("Arial", 11), fg="#bdc3c7", bg="#34495e")
        lbl_expr_prompt.pack(pady=(15, 5))
        
        self.expr_var = tk.StringVar()
        self.combo_expr = ttk.Combobox(master, textvariable=self.expr_var, values=EXPRESSIONS, state="readonly", font=("Arial", 11), width=25)
        self.combo_expr.pack(pady=5)
        self.combo_expr.current(0) 
        
        btn_launch = tk.Button(master, text="Click Here to Start", font=("Arial", 11, "bold"), bg="#27ae60", fg="white", width=28, height=2, command=self.confirm_selection)
        btn_launch.pack(pady=25)

    def confirm_selection(self):
        selected_group = self.group_var.get()
        selected_expr = self.expr_var.get()
        self.master.destroy()
        
        pipeline_root = tk.Tk()
        ProductionPipelineGUI(pipeline_root, selected_group, selected_expr)
        pipeline_root.mainloop()


if __name__ == "__main__":
    selector_root = tk.Tk()
    app = CohortSelectorWindow(selector_root)
    selector_root.mainloop()