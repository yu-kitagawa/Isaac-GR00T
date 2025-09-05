# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for Eagle2_5_VL.
copy from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_onevision/processing_llava_onevision.py
"""

import base64
import math
import os
import re
import time
import warnings
from functools import lru_cache
from io import BytesIO
from typing import Any, List, Literal, Optional, Union

import requests
import torch
import torchvision
from packaging import version
from PIL import Image
from torchvision import io
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, VideoInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging

logger = logging.get_logger(__name__)


FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 256


def adjust_by_factor(
    number: int, factor: int, method: Literal["round", "ceil", "floor"] = "round"
) -> int:
    """Adjusts 'number' to the nearest, ceiling, or floor multiple of 'factor'."""
    op = {"round": round, "ceil": math.ceil, "floor": math.floor}[method]
    return op(number / factor) * factor


def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == "RGBA":
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")


def fetch_image(ele: dict[str, str | Image.Image]) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        response = requests.get(image, stream=True)
        image_obj = Image.open(BytesIO(response.content))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(
            f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}"
        )
    image = to_rgb(image_obj)
    if "scale_factor" in ele:
        scale_factor = ele["scale_factor"]
        image = image.resize(
            (image.width * scale_factor, image.height * scale_factor), Image.BILINEAR
        )
    return image


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = adjust_by_factor(ele["nframes"], FRAME_FACTOR, method="round")
    else:
        fps = ele.get("fps", FPS)
        min_frames = adjust_by_factor(
            ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR, method="ceil"
        )
        max_frames = adjust_by_factor(
            ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR, method="floor"
        )
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = adjust_by_factor(nframes, FRAME_FACTOR, method="floor")
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(
            f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}."
        )
    return nframes


def _read_video_torchvision(
    ele: dict,
) -> (torch.Tensor, float, list):
    """read video using torchvision.io.read_video and return also per-frame timestamps"""
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn(
                "torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0."
            )
        if "file://" in video_path:
            video_path = video_path[7:]
    st = time.time()
    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    logger.info(
        f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s"
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    # Calculate frame indices and corresponding timestamps (based on video start time)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    start_time = ele.get("video_start", 0.0)
    timestamps = (start_time + idx.to(torch.float32) / video_fps).tolist()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = video[idx]
    return video, sample_fps, timestamps


def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None


def _read_video_decord(
    ele: dict,
) -> (torch.Tensor, float, list):
    """read video using decord.VideoReader and return also per-frame timestamps"""
    import decord

    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path)
    if "video_start" in ele or "video_end" in ele:
        raise NotImplementedError("not support start_pts and end_pts in decord for now.")
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    logger.info(
        f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s"
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    start_time = ele.get("video_start", 0.0)  # TODO:
    timestamps = [start_time + i / video_fps for i in idx]
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    return video, sample_fps, timestamps


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
}


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    return video_reader_backend


def fetch_video(
    ele: dict, return_video_sample_fps: bool = False
) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        video_reader_backend = get_video_reader_backend()
        try:
            video, sample_fps, timestamps = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        except Exception as e:
            logger.warning(
                f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}"
            )
            video, sample_fps, timestamps = VIDEO_READER_BACKENDS["torchvision"](ele)

        nframes, _, height, width = video.shape

        if return_video_sample_fps:
            return video, sample_fps, timestamps
        return video
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image({"image": video_element, **process_info}) for video_element in ele["video"]
        ]
        nframes = adjust_by_factor(len(images), FRAME_FACTOR, method="ceil")
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))

        timestamps = [-1 for i in range(nframes)]  # not sure about this
        if return_video_sample_fps:
            return images, process_info.pop("fps", 2.0), timestamps
        return images


class Eagle2_5_VLProcessorKwargs(ProcessingKwargs, total=False):
    # see processing_utils.ProcessingKwargs documentation for usage.
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {},
        "videos_kwargs": {"max_dynamic_tiles": 1},
    }


class Eagle2_5_VLProcessor(ProcessorMixin):
    r"""
    Constructs a Eagle2_5_VL processor which wraps a Eagle2_5_VL video processor, Eagle2_5_VL image processor and a Eagle2_5_VL tokenizer into a single processor.

    [`Eagle2_5_VLProcessor`] offers all the functionalities of [`Eagle2_5_VLVideoProcessor`], [`Eagle2_5_VLImageProcessor`] and [`Eagle2_5_VLTokenizer`]. See the
    [`~Eagle2_5_VLVideoProcessor.__call__`], [`~Eagle2_5_VLProcessor.__call__`] and [`~Eagle2_5_VLProcessor.decode`] for more information.

    Args:
        image_processor ([`LlavaOnevisionImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        num_image_tokens (`int`, *optional*):
            Number of image tokens for one imagethat will be returned by vision tower.
        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Shoudl be same as in model's config
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote image location.
        video_token (`str`, *optional*, defaults to `"<video>"`):
            Special token used to denote video location.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = [
        "chat_template",
        "num_image_tokens",
        "vision_feature_select_strategy",
        "image_token",
        "video_token",
        "images_kwargs",
        "videos_kwargs",
        "text_kwargs",
    ]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        vision_feature_select_strategy=None,
        chat_template=None,
        image_token="<IMG_CONTEXT>",
        video_token="<IMG_CONTEXT>",
        tokens_per_tile=256,
        image_placeholder="image",
        video_placeholder="video",
        image_start_token="<img>",
        image_end_token="</img>",
        **kwargs,
    ):
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_token = (
            tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        )
        self.video_token = (
            tokenizer.video_token if hasattr(tokenizer, "video_token") else video_token
        )
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        self.image_placeholder = image_placeholder
        self.video_placeholder = video_placeholder
        self.tokens_per_tile = tokens_per_tile
        self.image_start_token = image_start_token
        self.image_end_token = image_end_token
        if "auto_map" in kwargs:
            self.auto_map = kwargs["auto_map"]
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def replace_media_placeholder(
        self, text, image_list, video_list, timestamps_list, fps_list, **output_kwargs
    ):

        num_of_images_in_this_sample = 0
        num_of_videos_in_this_sample = 0
        # Regular expression pattern to match formats like <image-1> or <video-2>
        pattern = re.compile(rf"<({self.image_placeholder}|{self.video_placeholder})-(\d+)>")
        unified_frame_list = []

        # image_min_dynamic_tiles = output_kwargs["images_kwargs"].get(
        #     "min_dynamic_tiles", self.image_processor.min_dynamic_tiles
        # )
        # image_max_dynamic_tiles = output_kwargs["images_kwargs"].get(
        #     "max_dynamic_tiles", self.image_processor.max_dynamic_tiles
        # )
        # image_use_thumbnail = output_kwargs["images_kwargs"].get(
        #     "use_thumbnail", self.image_processor.use_thumbnail
        # )
        video_min_dynamic_tiles = output_kwargs["videos_kwargs"].get(
            "min_dynamic_tiles", self.image_processor.min_dynamic_tiles
        )
        video_max_dynamic_tiles = output_kwargs["videos_kwargs"].get(
            "max_dynamic_tiles", self.image_processor.max_dynamic_tiles
        )
        video_use_thumbnail = output_kwargs["videos_kwargs"].get(
            "use_thumbnail", self.image_processor.use_thumbnail
        )

        tile_size = self.image_processor.size.get("height", 448)

        # Function to replace tags in a single text
        def replace_in_text(text):
            # repl callback function for each match replacement operation
            def repl(match):
                nonlocal unified_frame_list
                nonlocal num_of_images_in_this_sample
                nonlocal num_of_videos_in_this_sample
                media_type = match.group(1)  # 'image' or 'video'
                idx_in_list = int(match.group(2)) - 1  # Convert to list index (0-based)
                # Select the corresponding path based on media type
                idx_mapper = {
                    0: "first",
                    1: "second",
                    2: "third",
                    3: "fourth",
                    4: "fifth",
                    5: "sixth",
                    6: "seventh",
                    7: "eighth",
                    8: "ninth",
                    9: "tenth",
                }
                if media_type == "image":
                    image_inputs = self.image_processor(
                        images=[image_list[idx_in_list]],
                        videos=None,
                        **output_kwargs["images_kwargs"],
                    )
                    num_all_tiles = image_inputs["pixel_values"].shape[0]
                    special_placeholder = f"<image {idx_in_list+1}>{self.image_start_token}{self.image_token * num_all_tiles * self.tokens_per_tile}{self.image_end_token}"
                    unified_frame_list.append(image_inputs)
                    num_of_images_in_this_sample += 1

                elif media_type == "video":
                    video_inputs = self.image_processor(
                        images=None,
                        videos=[video_list[idx_in_list]],
                        **output_kwargs["videos_kwargs"],
                    )
                    num_all_tiles = video_inputs["pixel_values"].shape[0]
                    image_sizes = video_inputs["image_sizes"]
                    if timestamps_list is not None and -1 not in timestamps_list:
                        frame_timestamps = timestamps_list[idx_in_list]
                    else:
                        frame_timestamps = None
                    sampled_fps = fps_list[idx_in_list] if fps_list is not None else None

                    num_of_tiles_each_frame = [
                        self.get_number_tiles_based_on_image_size(
                            image_size,
                            video_min_dynamic_tiles,
                            video_max_dynamic_tiles,
                            video_use_thumbnail,
                            tile_size,
                        )
                        for image_size in image_sizes
                    ]
                    assert (
                        sum(num_of_tiles_each_frame) == num_all_tiles
                    ), f"The number of tiles in each frame is not equal to the total number of tiles: {sum(num_of_tiles_each_frame)} != {num_all_tiles}"

                    if frame_timestamps is not None:
                        assert len(frame_timestamps) == len(
                            num_of_tiles_each_frame
                        ), f"The number of timestamps is not equal to the number of frames: {len(frame_timestamps)} != {len(num_of_tiles_each_frame)}"
                        special_placeholder = [
                            f"Frame {i+1} sample at {frame_timestamps[i]:.2f}s: {self.image_start_token}{self.image_token * num_of_tiles * self.tokens_per_tile}{self.image_end_token}"
                            for i, num_of_tiles in enumerate(num_of_tiles_each_frame)
                        ]
                    else:
                        special_placeholder = [
                            f"Frame {i+1}: {self.image_start_token}{self.image_token * num_of_tiles * self.tokens_per_tile}{self.image_end_token}"
                            for i, num_of_tiles in enumerate(num_of_tiles_each_frame)
                        ]

                    if sampled_fps is not None:
                        special_placeholder = (
                            f"The {idx_mapper[idx_in_list]} video sampled with {sampled_fps:.2f} fps: "
                            + "".join(special_placeholder)
                        )
                    else:
                        special_placeholder = f"The {idx_mapper[idx_in_list]} video: " + "".join(
                            special_placeholder
                        )
                    unified_frame_list.append(video_inputs)
                    num_of_videos_in_this_sample += 1
                else:
                    raise ValueError(f"Unknown media type: {media_type}")
                return special_placeholder

            return pattern.sub(repl, text)

        text = replace_in_text(text)
        if len(unified_frame_list) > 0:
            pixel_values = torch.cat([frame["pixel_values"] for frame in unified_frame_list])
            image_sizes = torch.cat([frame["image_sizes"] for frame in unified_frame_list])
        else:
            pixel_values = None
            image_sizes = None
        return (
            text,
            pixel_values,
            image_sizes,
            num_of_images_in_this_sample,
            num_of_videos_in_this_sample,
        )

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audio=None,
        videos: VideoInput = None,
        **kwargs: Unpack[Eagle2_5_VLProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        LlavaNextImageProcessor's [`~LlavaNextImageProcessor.__call__`] if `images` is not `None`. Please refer to the docstring
        of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of a video input to be fed to a model. Returned when `videos` is not `None`.
            - **image_sizes** -- Size of each image that will be used to unpad an image. Returned when `images` is not `None`.
        """

        output_kwargs = self._merge_kwargs(
            Eagle2_5_VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text_list = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")
        elif isinstance(text, list) and isinstance(text[0], str):
            text_list = text

        if images is None:
            images = []
        if videos is None:
            videos = []

        pixel_values_list = []
        image_sizes_list = []
        new_sample_list = []
        image_start_idx = 0
        video_start_idx = 0
        timestamps_batch = output_kwargs["videos_kwargs"].pop("timestamps", None)
        fps_batch = output_kwargs["videos_kwargs"].pop("fps", None)
        for sample in text_list:
            timestamps_list = (
                timestamps_batch[video_start_idx:] if timestamps_batch is not None else None
            )
            fps_list = fps_batch[video_start_idx:] if fps_batch is not None else None
            (
                sample,
                pixel_values,
                image_sizes,
                num_of_images_in_this_sample,
                num_of_videos_in_this_sample,
            ) = self.replace_media_placeholder(
                sample,
                images[image_start_idx:],
                videos[video_start_idx:],
                timestamps_list,
                fps_list,
                **output_kwargs,
            )
            new_sample_list.append(sample)
            if pixel_values is not None:
                pixel_values_list.append(pixel_values)
                image_sizes_list.append(image_sizes)
            image_start_idx += num_of_images_in_this_sample
            video_start_idx += num_of_videos_in_this_sample

        if len(pixel_values_list) > 0:
            image_inputs = {
                "pixel_values": torch.cat(pixel_values_list),
                "image_sizes": torch.cat(image_sizes_list),
            }
        else:
            image_inputs = {}
        video_inputs = {}
        text_inputs = self.tokenizer(new_sample_list, **output_kwargs["text_kwargs"])
        return BatchFeature(data={**text_inputs, **image_inputs, **video_inputs})

    def get_number_tiles_based_on_image_size(
        self, image_size: tuple, min_num: int, max_num: int, use_thumbnail: bool, tile_size: int
    ) -> int:
        """
        Get the number of tiles based on the image size.
        """
        orig_height, orig_width = image_size
        aspect_ratio = orig_width / orig_height
        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.image_processor.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, tile_size
        )
        tiles_num = target_aspect_ratio[0] * target_aspect_ratio[1]
        if use_thumbnail and tiles_num > 1:
            tiles_num += 1
        return tiles_num

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    # override to save video-config in a separate config file
    def save_pretrained(self, save_directory, **kwargs):
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)

        outputs = super().save_pretrained(save_directory, **kwargs)
        return outputs

    # override to load video-config from a separate config file
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # if return_unused_kwargs a tuple is returned where the second element is 'unused_kwargs'
        if isinstance(processor, tuple):
            processor = processor[0]
        return processor

    # Copy from https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
    def process_vision_info(
        self,
        conversations: list[dict] | list[list[dict]],
        return_video_kwargs: bool = False,
    ) -> tuple[
        list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]
    ]:

        vision_infos = self.extract_vision_info(conversations)
        ## Read images or videos
        image_inputs = []
        video_inputs = []
        video_sample_fps_list = []
        video_timestamps_list = []
        for vision_info in vision_infos:
            if "image" in vision_info or "image_url" in vision_info:
                image_inputs.append(fetch_image(vision_info))
            elif "video" in vision_info:
                video_input, video_sample_fps, video_timestamps = fetch_video(
                    vision_info, return_video_sample_fps=True
                )
                video_sample_fps_list.append(video_sample_fps)
                video_inputs.append(video_input)
                video_timestamps_list.append(video_timestamps)
            else:
                raise ValueError("image, image_url or video should in content.")
        if len(image_inputs) == 0:
            image_inputs = None
        if len(video_inputs) == 0:
            video_inputs = None
        if return_video_kwargs:
            return (
                image_inputs,
                video_inputs,
                {"fps": video_sample_fps_list, "timestamps": video_timestamps_list},
            )
        return image_inputs, video_inputs

    def extract_vision_info(self, conversations: list[dict] | list[list[dict]]) -> list[dict]:
        vision_infos = []
        if isinstance(conversations[0], dict):
            conversations = [conversations]
        for conversation in conversations:
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if (
                            "image" in ele
                            or "image_url" in ele
                            or "video" in ele
                            or ele["type"] in ("image", "image_url", "video")
                        ):
                            vision_infos.append(ele)
        return vision_infos

    def py_apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        """
        Renders a chat conversation using a custom template with verification of tokens.

        The purpose is to check for the existence of tokens like "<image-1>" or "<video-1>"
        in the message text and skip adding them if they already exist.

        Args:
            messages (list): A list of message dictionaries. Each message should contain:
                - 'role': The role of the speaker (e.g., 'system', 'user', 'assistant').
                - 'content': Either a string or a list of content blocks. In the list each block may contain:
                      * 'type': The type of content, such as 'image' or 'video'.
                      * 'text': The actual text if present.
                      * Other keys such as 'image', 'image_url', or 'video'.
            add_generation_prompt (bool): If True, appends "<|im_start|>assistant" at the end of the rendered string.
            tokenize (bool): If True, tokenize the rendered string.
        Returns:
            str: The final rendered chat string according to the specified template.
        """
        assert not tokenize, "tokenize is not supported yet"
        result = ""
        image_count = 0
        video_count = 0

        message_text = ""
        for idx, message in enumerate(messages):
            if message.get("role") != "user":
                continue
            # If content is a string, simply output it.
            content = message.get("content")
            if isinstance(content, str):
                message_text += content
            elif isinstance(content, list):
                # Process each content item.
                for item in content:
                    # If the block is a dictionary and contains text, add it to message_text.
                    if isinstance(item, dict) and "text" in item:
                        message_text += item["text"]
                    # If an item is already a string in the list, add it directly.
                    elif isinstance(item, str):
                        message_text += item

        for idx, message in enumerate(messages):
            # If the first message is not from the system, prepend a default system message.
            if idx == 0 and message.get("role") != "system":
                result += "<|im_start|>system\n"
                result += "You are a helpful assistant.\n"
                result += "<|im_end|>\n"

            # Start the current message block with its role.
            result += f"<|im_start|>{message.get('role', '')}\n"
            content = message.get("content")

            # If content is a string, simply output it.
            if isinstance(content, str):
                result += content
                result += "<|im_end|>\n"
            else:
                # Process each content item.
                for item in content:
                    # Check if the item is an image (explicitly by type or by key presence).
                    if isinstance(item, dict) and (
                        item.get("type") == "image" or "image" in item or "image_url" in item
                    ):
                        image_count += 1
                        candidate_token = f"<image-{image_count}>"
                        # Only add the token if it is not already present in the collected text.
                        if candidate_token not in message_text:
                            result += candidate_token
                    # Check if the item is a video.
                    elif isinstance(item, dict) and (
                        item.get("type") == "video" or "video" in item
                    ):
                        video_count += 1
                        candidate_token = f"<video-{video_count}>"
                        # Only add the token if it is not already present.
                        if candidate_token not in message_text:
                            result += candidate_token
                    # If the item contains text, add it.
                    elif isinstance(item, dict) and "text" in item:
                        result += item["text"]
                    # If the item is a string (and not handled already), add it.
                    elif isinstance(item, str):
                        result += item
                result += "<|im_end|>\n"

        # Optionally add assistant generation prompt at the end.
        if add_generation_prompt:
            result += "<|im_start|>assistant\n"

        return result

    @classmethod
    def from_args_and_dict(cls, args, processor_dict: dict[str, Any], **kwargs):
        """
        Instantiates a type of [`~processing_utils.ProcessingMixin`] from a Python dictionary of parameters.

        Args:
            processor_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the processor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~processing_utils.ProcessingMixin.to_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the processor object.

        Returns:
            [`~processing_utils.ProcessingMixin`]: The processor object instantiated from those
            parameters.
        """
        processor_dict = processor_dict.copy()
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        # We have to pop up some unused (but specific) kwargs and then validate that it doesn't contain unused kwargs
        # If we don't pop, some specific kwargs will raise a warning
        if "processor_class" in processor_dict:
            del processor_dict["processor_class"]

        # if "auto_map" in processor_dict:
        #    del processor_dict["auto_map"]

        unused_kwargs = cls.validate_init_kwargs(
            processor_config=processor_dict, valid_kwargs=cls.valid_kwargs
        )
        processor = cls(*args, **processor_dict)

        # Update processor with kwargs if needed
        for key in set(kwargs.keys()):
            if hasattr(processor, key):
                setattr(processor, key, kwargs.pop(key))

        kwargs.update(unused_kwargs)
        logger.info(f"Processor {processor}")
        if return_unused_kwargs:
            return processor, kwargs
        else:
            return processor


__all__ = ["Eagle2_5_VLProcessor"]
