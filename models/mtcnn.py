import torch
from torch import nn
import numpy as np
import os
from pathlib import Path

from .alignment import alignment
from .utils.detect_face import detect_face, extract_face
from .utils.download import download_url_to_file
from torchvision.transforms.functional import to_pil_image as tensor_to_pil_image


class PNet(nn.Module):
    """MTCNN PNet.
    
    Keyword Arguments:
        pretrained {bool} -- Whether or not to load saved pretrained weights (default: {True})
    """
    PRETRAINED_URL = "https://raw.githubusercontent.com/Redbeard-himalaya/facenet-pytorch/v2.5.3-dev/data/pnet.pt"

    def __init__(self, pretrained: Path = None, progress: bool = True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3)
        self.prelu2 = nn.PReLU(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.prelu3 = nn.PReLU(32)
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1)
        self.softmax4_1 = nn.Softmax(dim=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)

        self.training = False

        if pretrained:
            if not pretrained.exists():
                pretrained.parent.mkdir(parents=True, exist_ok=True)
                download_url_to_file(self.PRETRAINED_URL, str(pretrained), progress=progress)
            state_dict = torch.load(str(pretrained), weights_only=True)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        a = self.conv4_1(x)
        a = self.softmax4_1(a)
        b = self.conv4_2(x)
        return b, a


class RNet(nn.Module):
    """MTCNN RNet.
    
    Keyword Arguments:
        pretrained {bool} -- Whether or not to load saved pretrained weights (default: {True})
    """
    PRETRAINED_URL = "https://raw.githubusercontent.com/Redbeard-himalaya/facenet-pytorch/v2.5.3-dev/data/rnet.pt"

    def __init__(self, pretrained: Path = None, progress: bool = True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 28, kernel_size=3)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2)
        self.prelu3 = nn.PReLU(64)
        self.dense4 = nn.Linear(576, 128)
        self.prelu4 = nn.PReLU(128)
        self.dense5_1 = nn.Linear(128, 2)
        self.softmax5_1 = nn.Softmax(dim=1)
        self.dense5_2 = nn.Linear(128, 4)

        self.training = False

        if pretrained:
            if not pretrained.exists():
                pretrained.parent.mkdir(parents=True, exist_ok=True)
                download_url_to_file(self.PRETRAINED_URL, str(pretrained), progress=progress)
            state_dict = torch.load(str(pretrained), weights_only=True)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense4(x.view(x.shape[0], -1))
        x = self.prelu4(x)
        a = self.dense5_1(x)
        a = self.softmax5_1(a)
        b = self.dense5_2(x)
        return b, a


class ONet(nn.Module):
    """MTCNN ONet.
    
    Keyword Arguments:
        pretrained {bool} -- Whether or not to load saved pretrained weights (default: {True})
    """
    PRETRAINED_URL = "https://raw.githubusercontent.com/Redbeard-himalaya/facenet-pytorch/v2.5.3-dev/data/onet.pt"

    def __init__(self, pretrained: Path = None, progress: bool = True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
        self.prelu4 = nn.PReLU(128)
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(256)
        self.dense6_1 = nn.Linear(256, 2)
        self.softmax6_1 = nn.Softmax(dim=1)
        self.dense6_2 = nn.Linear(256, 4)
        self.dense6_3 = nn.Linear(256, 10)

        self.training = False

        if pretrained:
            if not pretrained.exists():
                pretrained.parent.mkdir(parents=True, exist_ok=True)
                download_url_to_file(self.PRETRAINED_URL, str(pretrained), progress=progress)
            state_dict = torch.load(str(pretrained), weights_only=True)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense5(x.view(x.shape[0], -1))
        x = self.prelu5(x)
        a = self.dense6_1(x)
        a = self.softmax6_1(a)
        b = self.dense6_2(x)
        c = self.dense6_3(x)
        return b, c, a


class MTCNN(nn.Module):
    """MTCNN face detection module.

    This class loads pretrained P-, R-, and O-nets and returns images cropped to include the face
    only, given raw input images of one of the following types:
        - PIL image or list of PIL images
        - numpy.ndarray (uint8) representing either a single image (3D) or a batch of images (4D).
    Cropped faces can optionally be saved to file
    also.
    
    Keyword Arguments:
        image_size {int} -- Output image size in pixels. The image will be square. (default: {160})
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image. 
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size (this is a bug in davidsandberg/facenet).
            (default: {0})
        min_face_size {int} -- Minimum face size to search for. (default: {20})
        thresholds {list} -- MTCNN face detection thresholds (default: {[0.6, 0.7, 0.7]})
        factor {float} -- Factor used to create a scaling pyramid of face sizes. (default: {0.709})
        post_process {bool} -- Whether or not to post process images tensors before returning.
            (default: {True})
        select_largest {bool} -- If True, if multiple faces are detected, the largest is returned.
            If False, the face with the highest detection probability is returned.
            (default: {True})
        selection_method {string} -- Which heuristic to use for selection. Default None. If
            specified, will override select_largest:
                    "probability": highest probability selected
                    "largest": largest box selected
                    "largest_over_threshold": largest box over a certain probability selected
                    "center_weighted_size": box size minus weighted squared offset from image center
                (default: {None})
        keep_all {bool} -- If True, all detected faces are returned, in the order dictated by the
            select_largest parameter. If a save_path is specified, the first face is saved to that
            path and the remaining faces are saved to <save_path>1, <save_path>2 etc.
            (default: {False})
        device {torch.device} -- The device on which to run neural net passes. Image tensors and
            models are copied to this device before running forward passes. (default: {None})
    """

    def __init__(
        self, image_size=160, margin=0, min_face_size=20, model_dir: Path = None,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, align=True,
        select_largest=True, selection_method=None, keep_all=False, device=None, progress=True,
    ):
        super().__init__()

        self.image_size = image_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.factor = factor
        self.post_process = post_process
        self.align = align
        self.select_largest = select_largest
        self.keep_all = keep_all
        self.selection_method = selection_method

        if model_dir is None:
            model_dir = Path.home() / ".face_search"
        self.pnet = PNet(pretrained=model_dir / "1", progress=progress)
        self.rnet = RNet(pretrained=model_dir / "2", progress=progress)
        self.onet = ONet(pretrained=model_dir / "3", progress=progress)

        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
            self.to(device)

        if not self.selection_method:
            self.selection_method = 'largest' if self.select_largest else 'probability'

    def forward(self,
                img,
                save_path=None,
                return_prob=False,
                return_box=False,
                return_point=False,
    ):
        """Run MTCNN face detection on a PIL image or numpy array. This method performs both
        detection and extraction of faces, returning tensors representing detected faces rather
        than the bounding boxes. To access bounding boxes, see the MTCNN.detect() method below.
        
        Arguments:
            img {PIL.Image, np.ndarray, or list} -- A PIL image, np.ndarray, torch.Tensor (values are not scaled), or list.
        
        Keyword Arguments:
            save_path {str} or {List[str]} -- An optional save path for the cropped image.
                Note that when self.post_process=True, although the returned tensor is post
                 processed, the saved face image is not, so it is a true representation of the
                 face in the input image.
                If `img` is a list of images, `save_path` should be a list of equal length.
                (default: {None})
            return_prob {bool} -- Whether or not to return the detection probability.
                (default: {False})
        
        Returns:
            Union[torch.Tensor, tuple(torch.tensor, float)] -- If detected, cropped image of a face
                with dimensions 3 x image_size x image_size. Optionally, the probability that a
                face was detected. If self.keep_all is True, n detected faces are returned in an
                n x 3 x image_size x image_size tensor with an optional list of detection
                probabilities. If `img` is a list of images, the item(s) returned have an extra 
                dimension (batch) as the first dimension.

        Example:
        >>> from facenet_pytorch import MTCNN
        >>> mtcnn = MTCNN()
        >>> face_tensor, prob = mtcnn(img, save_path='face.png', return_prob=True)
        """

        # create tensor view from (C x H x W) to (H x W x C)
        if isinstance(img, torch.Tensor):
            dim_len = len(img.shape)
            if dim_len == 3:
                img = img.permute(1, 2, 0)
            elif dim_len == 4:
                img = img.permute(0, 2, 3, 1)
            else:
                raise TypeError(f"torch img dim length {dim_len} is not 3 or 4")

        # Detect faces
        batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
        # Select faces
        if not self.keep_all:
            batch_boxes, batch_probs, batch_points = self.select_boxes(
                batch_boxes, batch_probs, batch_points, img, method=self.selection_method
            )
        # Extract faces
        faces = self.extract(img, batch_boxes, None)

        # faces are faces of a batch of images, each element of faces contains detected faces
        # of a signle image. batch_points are points of each faces
        # version 1: avg cost 0.350s per 500 heads
        # if self.align:
        #     aligned_faces = []
        #     #import pdb; pdb.set_trace()
        #     for img_faces, img_points in zip(faces, batch_points):
        #         if img_faces is None:
        #             aligned_faces.append(None)
        #         else:
        #             img_aligned_faces = []
        #             for face, point in zip(img_faces, img_points):
        #                 aligned_face, _, _ = alignment(face, point)
        #                 img_aligned_faces.append(aligned_face)
        #             aligned_faces.append(torch.stack(img_aligned_faces))
        #     faces = aligned_faces

        # version 2: avg cost 0.170s per 500 heads
        if self.align:
            # create batch data
            face_batch = []
            landmark_batch = []
            for img_faces, img_points in zip(faces, batch_points):
                if img_faces is None:
                    continue
                face_batch.append(img_faces)
                landmark_batch.append(img_points.astype(np.float32))
            if len(face_batch) > 0:
                face_batch = torch.cat(face_batch, axis=0)
                landmark_batch = torch.tensor(np.concatenate(landmark_batch, axis=0)).to(self.device)
                aligned_faces_batch = alignment(face_batch, landmark_batch)
            aligned_faces = []
            start_idx = 0
            for img_faces in faces:
                if img_faces is None:
                    aligned_faces.append(None)
                else:
                    size = len(img_faces)
                    aligned_faces.append(aligned_faces_batch[start_idx : start_idx+size])
                    start_idx += size
            faces = aligned_faces

        # version 3: avg cost 0.200s per 500 heads
        # if self.align:
        #     # create batch data
        #     none_posits = []
        #     for i, img_points in enumerate(batch_points):
        #         if img_points is None:
        #             none_posits.append(i)
        #             faces[i] = torch.empty(1,3,160,160)
        #             batch_points[i] = np.zeros((1,5,2))
        #     face_batch = torch.cat(faces, axis=0)
        #     landmark_batch = np.concatenate(batch_points, axis=0)
        #     aligned_faces_batch, _ = alignment(face_batch, landmark_batch)
        #     aligned_faces = []
        #     none_idx = 0
        #     start_idx = 0
        #     none_posits_len = len(none_posits)
        #     for i, img_faces in enumerate(faces):
        #         size = len(img_faces)
        #         if none_idx < none_posits_len and i == none_posits[none_idx]:
        #             aligned_faces.append(None)
        #             none_idx += 1
        #         else:
        #             aligned_faces.append(aligned_faces_batch[start_idx : start_idx+size])
        #         start_idx += size
        #     faces = aligned_faces

        if save_path:
            if isinstance(save_path, str):
                save_path = [save_path]
            for imgs, save_path in zip(faces, save_path):
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                if imgs is None:
                    continue
                for i, im in enumerate(imgs):
                    if self.post_process:
                        im = im * 128.0 + 127.5
                    if i == 0:
                        _save_path = save_path
                    else:
                        _save_path = save_path.parent/f"{save_path.stem}_{i}{save_path.suffix}"
                    tensor_to_pil_image(im.to(dtype=torch.uint8), "RGB").save(_save_path)

        ret = (faces,)
        if return_prob:
            ret += (batch_probs,)
        if return_box:
            ret += (batch_boxes,)
        if return_point:
            ret += (batch_points,)
        return ret

    def detect(self, img, landmarks=False):
        """Detect all faces in PIL image and return bounding boxes and optional facial landmarks.

        This method is used by the forward method and is also useful for face detection tasks
        that require lower-level handling of bounding boxes and facial landmarks (e.g., face
        tracking). The functionality of the forward function can be emulated by using this method
        followed by the extract_face() function.
        
        Arguments:
            img {PIL.Image, np.ndarray, or list} -- A PIL image, np.ndarray, torch.Tensor, or list.

        Keyword Arguments:
            landmarks {bool} -- Whether to return facial landmarks in addition to bounding boxes.
                (default: {False})
        
        Returns:
            tuple(numpy.ndarray, list) -- For N detected faces, a tuple containing an
                Nx4 array of bounding boxes and a length N list of detection probabilities.
                Returned boxes will be sorted in descending order by detection probability if
                self.select_largest=False, otherwise the largest face will be returned first.
                If `img` is a list of images, the items returned have an extra dimension
                (batch) as the first dimension. Optionally, a third item, the facial landmarks,
                are returned if `landmarks=True`.

        Example:
        >>> from PIL import Image, ImageDraw
        >>> from facenet_pytorch import MTCNN, extract_face
        >>> mtcnn = MTCNN(keep_all=True)
        >>> boxes, probs, points = mtcnn.detect(img, landmarks=True)
        >>> # Draw boxes and save faces
        >>> img_draw = img.copy()
        >>> draw = ImageDraw.Draw(img_draw)
        >>> for i, (box, point) in enumerate(zip(boxes, points)):
        ...     draw.rectangle(box.tolist(), width=5)
        ...     for p in point:
        ...         draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
        ...     extract_face(img, box, save_path='detected_face_{}.png'.format(i))
        >>> img_draw.save('annotated_faces.png')
        """

        with torch.no_grad():
            batch_boxes, batch_points = detect_face(
                img, self.min_face_size,
                self.pnet, self.rnet, self.onet,
                self.thresholds, self.factor,
                self.device
            )

        boxes, probs, points = [], [], []
        for box, point in zip(batch_boxes, batch_points):
            box = np.array(box)
            point = np.array(point)
            if len(box) == 0:
                boxes.append(None)
                probs.append(None)
                points.append(None)
            elif self.select_largest:
                box_order = np.argsort((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]))[::-1]
                box = box[box_order]
                point = point[box_order]
                boxes.append(box[:, :4])
                probs.append(box[:, 4])
                points.append(point)
            else:
                boxes.append(box[:, :4])
                probs.append(box[:, 4])
                points.append(point)
        boxes = np.array(boxes, dtype=object)
        probs = np.array(probs, dtype=object)
        points = np.array(points, dtype=object)

        if landmarks:
            return boxes, probs, points

        return boxes, probs

    def select_boxes(
        self, all_boxes, all_probs, all_points, imgs, method='probability', threshold=0.9,
        center_weight=2.0
    ):
        """Selects a single box from multiple for a given image using one of multiple heuristics.

        Arguments:
                all_boxes {np.ndarray} -- Ix0 ndarray where each element is a Nx4 ndarry of
                    bounding boxes for N detected faces in I images (output from self.detect).
                all_probs {np.ndarray} -- Ix0 ndarray where each element is a Nx0 ndarry of
                    probabilities for N detected faces in I images (output from self.detect).
                all_points {np.ndarray} -- Ix0 ndarray where each element is a Nx5x2 array of
                    points for N detected faces. (output from self.detect).
                imgs {PIL.Image, np.ndarray, or list} -- A PIL image, np.ndarray, torch.Tensor, or list.

        Keyword Arguments:
                method {str} -- Which heuristic to use for selection:
                    "probability": highest probability selected
                    "largest": largest box selected
                    "largest_over_theshold": largest box over a certain probability selected
                    "center_weighted_size": box size minus weighted squared offset from image center
                    (default: {'probability'})
                threshold {float} -- theshold for "largest_over_threshold" method. (default: {0.9})
                center_weight {float} -- weight for squared offset in center weighted size method.
                    (default: {2.0})

        Returns:
                tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray) -- nx4 ndarray of bounding boxes
                    for n images. Ix0 array of probabilities for each box, array of landmark points.
        """

        #copying batch detection from extract, but would be easier to ensure detect creates consistent output.
        if (
                not isinstance(imgs, (list, tuple)) and
                not (isinstance(imgs, np.ndarray) and len(imgs.shape) == 4) and
                not (isinstance(imgs, torch.Tensor) and len(imgs.shape) == 4)
        ):
            imgs = [imgs]

        selected_boxes, selected_probs, selected_points = [], [], []
        for boxes, points, probs, img in zip(all_boxes, all_points, all_probs, imgs):
            
            if boxes is None:
                selected_boxes.append(None)
                selected_probs.append(None)
                selected_points.append(None)
                continue
            
            # If at least 1 box found
            boxes = np.array(boxes)
            probs = np.array(probs)
            points = np.array(points)
                
            if method == 'largest':
                box_order = np.argsort((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))[::-1]
            elif method == 'probability':
                box_order = np.argsort(probs)[::-1]
            elif method == 'center_weighted_size':
                box_sizes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                img_center = (img.width / 2, img.height/2)
                box_centers = np.array(list(zip((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2)))
                offsets = box_centers - img_center
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 1)
                box_order = np.argsort(box_sizes - offset_dist_squared * center_weight)[::-1]
            elif method == 'largest_over_threshold':
                box_mask = probs > threshold
                boxes = boxes[box_mask]
                box_order = np.argsort((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))[::-1]
                if sum(box_mask) == 0:
                    selected_boxes.append(None)
                    selected_probs.append([None])
                    selected_points.append(None)
                    continue

            box = boxes[box_order][[0]]
            prob = probs[box_order][[0]]
            point = points[box_order][[0]]
            selected_boxes.append(box)
            selected_probs.append(prob)
            selected_points.append(point)

        selected_boxes = np.array(selected_boxes, dtype=object)
        selected_probs = np.array(selected_probs, dtype=object)
        selected_points = np.array(selected_points, dtype=object)

        return selected_boxes, selected_probs, selected_points

    def extract(self, img, batch_boxes, save_path):
        # Determine if a batch or single image was passed
        if (
                not isinstance(img, (list, tuple)) and
                not (isinstance(img, np.ndarray) and len(img.shape) == 4) and
                not (isinstance(img, torch.Tensor) and len(img.shape) == 4)
        ):
            img = [img]

        # Parse save path(s)
        if save_path is not None:
            if isinstance(save_path, str):
                save_path = [save_path]
        else:
            save_path = [None for _ in range(len(img))]

        # Process all bounding boxes
        faces = []
        for im, box_im, path_im in zip(img, batch_boxes, save_path):
            if box_im is None:
                faces.append(None)
                continue

            if not self.keep_all:
                box_im = box_im[[0]]

            faces_im = []
            for i, box in enumerate(box_im):
                face_path = path_im
                if path_im is not None and i > 0:
                    save_name, ext = os.path.splitext(path_im)
                    face_path = save_name + '_' + str(i + 1) + ext

                face = extract_face(im, box, self.image_size, self.margin, face_path)
                if self.post_process:
                    face = fixed_image_standardization(face)
                faces_im.append(face)

            faces.append(torch.stack(faces_im))

        return faces


def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) * 0.0078125
    return processed_tensor


def prewhiten(x):
    mean = x.mean()
    std = x.std()
    std_adj = std.clamp(min=1.0/(float(x.numel())**0.5))
    y = (x - mean) / std_adj
    return y

