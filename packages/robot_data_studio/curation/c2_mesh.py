from __future__ import annotations

import math
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from pydantic import BaseModel, Field, model_validator

from robot_data_studio.quality.contracts import (
    QWEN_CORE_DEFAULT_WEIGHTS,
    QWEN_CORE_LABELS,
    MetricEvidence,
)
from robot_data_studio.quality.qwen_filters import _motion_transform, _transform_from_xyz_rpy

from .sam3 import Sam3Result
from .gr3_fk import _urdf_path
from .profile import KinematicsProfile, ReprojectionProfile


@dataclass(frozen=True)
class TriangleMesh:
    vertices: np.ndarray
    faces: np.ndarray
    link: str


@dataclass(frozen=True)
class MeshRenderResult:
    mask: np.ndarray
    depth: np.ndarray
    link_index: np.ndarray
    visual_count: int
    renderer: str = "cpu_triangle_zbuffer_v1"
    triangle_count: int = 0
    rendered_links: tuple[str, ...] = ()


class C2Calibration(BaseModel):
    version: str = Field(min_length=1)
    camera_id: str = Field(min_length=1)
    embodiment: str = Field(min_length=1)
    mask_checkpoint_kind: Literal["official_sam3", "fine_tuned_sam3"]
    calibrated: bool = False
    iou_review_threshold: float = Field(ge=0.0, le=1.0)
    threshold_version: str | None = None

    @model_validator(mode="after")
    def exact_claim_requires_assets(self) -> C2Calibration:
        if self.calibrated and self.mask_checkpoint_kind != "fine_tuned_sam3":
            raise ValueError("calibrated C2 requires the fine-tuned SAM3 checkpoint")
        if self.calibrated and not self.threshold_version:
            raise ValueError("calibrated C2 requires a versioned threshold")
        return self


def _box(size: np.ndarray, link: str) -> TriangleMesh:
    x, y, z = size / 2.0
    vertices = np.asarray(
        [[a, b, c] for a in (-x, x) for b in (-y, y) for c in (-z, z)],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
            [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
            [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
        ],
        dtype=np.int32,
    )
    return TriangleMesh(vertices, faces, link)


def _sphere(radius: float, link: str) -> TriangleMesh:
    vertices = radius * np.asarray(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float64,
    )
    faces = np.asarray(
        [[4, 0, 2], [4, 2, 1], [4, 1, 3], [4, 3, 0], [5, 2, 0], [5, 1, 2], [5, 3, 1], [5, 0, 3]],
        dtype=np.int32,
    )
    return TriangleMesh(vertices, faces, link)


def _cylinder(radius: float, length: float, link: str, segments: int = 20) -> TriangleMesh:
    vertices = []
    for z in (-length / 2.0, length / 2.0):
        vertices.extend(
            (radius * math.cos(2 * math.pi * i / segments), radius * math.sin(2 * math.pi * i / segments), z)
            for i in range(segments)
        )
    vertices.extend([(0.0, 0.0, -length / 2.0), (0.0, 0.0, length / 2.0)])
    faces = []
    for i in range(segments):
        nxt = (i + 1) % segments
        faces.extend(
            [[i, nxt, segments + nxt], [i, segments + nxt, segments + i], [2 * segments, nxt, i], [2 * segments + 1, segments + i, segments + nxt]]
        )
    return TriangleMesh(np.asarray(vertices), np.asarray(faces, dtype=np.int32), link)


@lru_cache(maxsize=256)
def _load_stl(path: Path, link: str) -> TriangleMesh:
    data = path.read_bytes()
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        if 84 + count * 50 == len(data):
            vertices: list[tuple[float, float, float]] = []
            faces = []
            for index in range(count):
                offset = 84 + index * 50 + 12
                triangle = struct.unpack_from("<9f", data, offset)
                base = len(vertices)
                vertices.extend(zip(triangle[0::3], triangle[1::3], triangle[2::3], strict=True))
                faces.append((base, base + 1, base + 2))
            return TriangleMesh(np.asarray(vertices), np.asarray(faces, dtype=np.int32), link)
    vertices = []
    for raw in data.decode("utf-8", errors="strict").splitlines():
        fields = raw.strip().split()
        if len(fields) == 4 and fields[0] == "vertex":
            vertices.append(tuple(float(value) for value in fields[1:]))
    if not vertices or len(vertices) % 3:
        raise ValueError(f"unsupported or corrupt STL visual: {path}")
    return TriangleMesh(
        np.asarray(vertices, dtype=np.float64),
        np.arange(len(vertices), dtype=np.int32).reshape(-1, 3),
        link,
    )


def _mesh_path(urdf_path: Path, filename: str) -> Path:
    if filename.startswith("package://"):
        relative = filename.removeprefix("package://").split("/", 1)[-1]
        candidates = [urdf_path.parent / relative, urdf_path.parent.parent / relative]
    else:
        candidates = [urdf_path.parent / filename]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise ValueError(f"URDF visual mesh is missing: {filename}")


def _visual_mesh(geometry: ET.Element, urdf_path: Path, link: str) -> TriangleMesh:
    box = geometry.find("box")
    sphere = geometry.find("sphere")
    cylinder = geometry.find("cylinder")
    mesh = geometry.find("mesh")
    if box is not None:
        return _box(np.fromstring(box.attrib["size"], sep=" "), link)
    if sphere is not None:
        return _sphere(float(sphere.attrib["radius"]), link)
    if cylinder is not None:
        return _cylinder(float(cylinder.attrib["radius"]), float(cylinder.attrib["length"]), link)
    if mesh is None:
        raise ValueError(f"unsupported visual geometry on {link}")
    path = _mesh_path(urdf_path, mesh.attrib["filename"])
    if path.suffix.casefold() != ".stl":
        raise ValueError(f"unsupported visual mesh format: {path.suffix}")
    loaded = _load_stl(path, link)
    scale = np.fromstring(mesh.attrib.get("scale", "1 1 1"), sep=" ")
    if scale.shape != (3,):
        raise ValueError("URDF mesh scale must have three values")
    return TriangleMesh(loaded.vertices * scale, loaded.faces, link)


def _joint_link_transforms(
    root: ET.Element,
    root_link: str,
    joint_state: dict[str, float],
    *,
    explicit_unrecorded_joint_positions: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Resolve a URDF tree without silently inventing movable-joint values.

    URDF mimic joints are derived from their named driver. Any other unrecorded
    movable joint must be explicitly frozen by the caller/profile.
    """

    explicit_positions = explicit_unrecorded_joint_positions or {}
    children: dict[str, list[tuple[ET.Element, str]]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        children.setdefault(parent.attrib["link"], []).append((joint, child.attrib["link"]))
    output = {root_link: np.eye(4, dtype=np.float64)}
    stack = [root_link]
    while stack:
        parent_link = stack.pop()
        for joint, child_link in children.get(parent_link, []):
            origin = joint.find("origin")
            xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            transform = output[parent_link] @ _transform_from_xyz_rpy(xyz, rpy)
            joint_type = joint.attrib.get("type", "fixed")
            if joint_type in {"revolute", "continuous", "prismatic"}:
                name = joint.attrib.get("name", "")
                mimic = joint.find("mimic")
                if mimic is not None:
                    driver = mimic.attrib.get("joint", "")
                    if driver not in joint_state:
                        raise ValueError(
                            f"mimic joint {name!r} driver {driver!r} is not recorded"
                        )
                    multiplier = float(mimic.attrib.get("multiplier", "1"))
                    offset = float(mimic.attrib.get("offset", "0"))
                    position = multiplier * float(joint_state[driver]) + offset
                elif name in joint_state:
                    position = float(joint_state[name])
                elif name in explicit_positions:
                    position = float(explicit_positions[name])
                else:
                    raise ValueError(f"recorded joint state missing URDF joint {name!r}")
                if not math.isfinite(position):
                    raise ValueError(f"URDF joint {name!r} position is non-finite")
                axis_node = joint.find("axis")
                axis = np.fromstring(axis_node.attrib.get("xyz", "1 0 0"), sep=" ") if axis_node is not None else np.asarray([1.0, 0.0, 0.0])
                transform = transform @ _motion_transform(joint_type, axis, position)
            output[child_link] = transform
            stack.append(child_link)
    return output


def _transform_vertices(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([vertices, np.ones((len(vertices), 1))], axis=1)
    return (transform @ homogeneous.T).T[:, :3]


def _project_camera_points(
    points: np.ndarray,
    *,
    intrinsic: np.ndarray,
    distortion_coefficients: np.ndarray | None,
    display_transform: np.ndarray | None,
) -> np.ndarray:
    """Project Camera0 points using the frozen OpenCV lens/display contract."""

    camera_points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    k = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    if distortion_coefficients is None:
        projected = (k @ camera_points.T).T
        xy = projected[:, :2] / projected[:, 2:3]
    else:
        import cv2

        distortion = np.asarray(distortion_coefficients, dtype=np.float64).reshape(-1)
        if distortion.shape != (14,):
            raise ValueError("OpenCV Camera0 distortion must contain exactly 14 values")
        xy, _ = cv2.projectPoints(
            camera_points,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            k,
            distortion,
        )
        xy = xy.reshape(-1, 2)
    if display_transform is not None:
        display = np.asarray(display_transform, dtype=np.float64).reshape(3, 3)
        homogeneous = np.concatenate([xy, np.ones((len(xy), 1))], axis=1)
        displayed = (display @ homogeneous.T).T
        xy = displayed[:, :2] / displayed[:, 2:3]
    return xy


def _rasterize(
    meshes: Sequence[TriangleMesh],
    camera_from_links: Sequence[np.ndarray],
    intrinsic: np.ndarray,
    image_shape: tuple[int, int],
    distortion_coefficients: np.ndarray | None = None,
    display_transform: np.ndarray | None = None,
) -> MeshRenderResult:
    height, width = image_shape
    depth = np.full((height, width), np.inf, dtype=np.float64)
    link_index = np.full((height, width), -1, dtype=np.int32)
    k = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    for mesh_index, (mesh, transform) in enumerate(zip(meshes, camera_from_links, strict=True)):
        vertices = _transform_vertices(mesh.vertices, transform)
        for face in mesh.faces:
            triangle = vertices[np.asarray(face, dtype=np.int32)]
            if np.any(triangle[:, 2] <= 1e-6) or not np.all(np.isfinite(triangle)):
                continue
            xy = _project_camera_points(
                triangle,
                intrinsic=k,
                distortion_coefficients=distortion_coefficients,
                display_transform=display_transform,
            )
            x0 = max(0, int(math.floor(float(np.min(xy[:, 0])))))
            x1 = min(width - 1, int(math.ceil(float(np.max(xy[:, 0])))))
            y0 = max(0, int(math.floor(float(np.min(xy[:, 1])))))
            y1 = min(height - 1, int(math.ceil(float(np.max(xy[:, 1])))))
            if x0 > x1 or y0 > y1:
                continue
            a, b, c = xy
            denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(denominator) <= 1e-12:
                continue
            yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
            px = xx + 0.5
            py = yy + 0.5
            w0 = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / denominator
            w1 = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / denominator
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
            inverse_z = (
                w0 / triangle[0, 2]
                + w1 / triangle[1, 2]
                + w2 / triangle[2, 2]
            )
            z = np.divide(
                1.0,
                inverse_z,
                out=np.full_like(inverse_z, np.inf),
                where=inverse_z > 0,
            )
            target = depth[y0 : y1 + 1, x0 : x1 + 1]
            visible = inside & (z < target)
            target[visible] = z[visible]
            link_index[y0 : y1 + 1, x0 : x1 + 1][visible] = mesh_index
    return MeshRenderResult(
        mask=np.isfinite(depth),
        depth=depth,
        link_index=link_index,
        visual_count=len(meshes),
        triangle_count=sum(len(mesh.faces) for mesh in meshes),
        rendered_links=tuple(mesh.link for mesh in meshes),
    )


@lru_cache(maxsize=4)
def _opengl_runtime(width: int, height: int):
    try:
        import moderngl
    except ImportError as error:
        raise RuntimeError(
            "opengl_zbuffer requires the optional moderngl renderer dependency"
        ) from error
    context = moderngl.create_standalone_context(backend="egl")
    program = context.program(
        vertex_shader="""
            #version 330
            in vec3 in_position;
            void main() {
                gl_Position = vec4(in_position.xy, in_position.z * 2.0 - 1.0, 1.0);
            }
        """,
        fragment_shader="""
            #version 330
            out float mask_value;
            void main() { mask_value = 1.0; }
        """,
    )
    color = context.texture((width, height), components=1, dtype="f1")
    depth = context.depth_texture((width, height))
    framebuffer = context.framebuffer(color_attachments=[color], depth_attachment=depth)
    return context, program, framebuffer, color, depth


def _rasterize_opengl(
    meshes: Sequence[TriangleMesh],
    camera_from_links: Sequence[np.ndarray],
    intrinsic: np.ndarray,
    image_shape: tuple[int, int],
    *,
    distortion_coefficients: np.ndarray | None,
    display_transform: np.ndarray | None,
) -> MeshRenderResult:
    """Rasterize every raw URDF triangle with an EGL depth buffer.

    Projection is computed through OpenCV first so its 14D lens model and the
    encoded-to-display affine remain identical to the frozen camera authority.
    No mesh decimation is performed.
    """

    import moderngl

    height, width = image_shape
    near = 1e-4
    triangles: list[np.ndarray] = []
    triangle_count = 0
    for mesh, transform in zip(meshes, camera_from_links, strict=True):
        vertices = _transform_vertices(mesh.vertices, transform)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        face_z = vertices[faces, 2]
        valid = np.all(np.isfinite(vertices[faces]), axis=(1, 2)) & np.all(
            face_z > near, axis=1
        )
        faces = faces[valid]
        if not len(faces):
            continue
        xy = _project_camera_points(
            vertices,
            intrinsic=intrinsic,
            distortion_coefficients=distortion_coefficients,
            display_transform=display_transform,
        )
        selected_xy = xy[faces]
        selected_z = vertices[faces, 2]
        positions = np.empty((len(faces), 3, 3), dtype=np.float32)
        positions[..., 0] = 2.0 * (selected_xy[..., 0] + 0.5) / width - 1.0
        positions[..., 1] = 1.0 - 2.0 * (selected_xy[..., 1] + 0.5) / height
        positions[..., 2] = 1.0 - near / selected_z
        finite = np.all(np.isfinite(positions), axis=(1, 2))
        if np.any(finite):
            triangles.append(np.ascontiguousarray(positions[finite].reshape(-1, 3)))
            triangle_count += int(np.count_nonzero(finite))
    if not triangles:
        return MeshRenderResult(
            mask=np.zeros(image_shape, dtype=bool),
            depth=np.full(image_shape, np.inf, dtype=np.float64),
            link_index=np.full(image_shape, -1, dtype=np.int32),
            visual_count=len(meshes),
            renderer="opengl_triangle_zbuffer_v1",
            triangle_count=0,
            rendered_links=tuple(mesh.link for mesh in meshes),
        )
    positions = np.concatenate(triangles, axis=0)
    context, program, framebuffer, color, depth_texture = _opengl_runtime(width, height)
    framebuffer.use()
    framebuffer.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
    context.enable(moderngl.DEPTH_TEST)
    context.disable(moderngl.CULL_FACE)
    buffer = context.buffer(positions.tobytes())
    vertex_array = context.simple_vertex_array(program, buffer, "in_position")
    try:
        vertex_array.render(mode=moderngl.TRIANGLES)
        raw_mask = np.frombuffer(color.read(alignment=1), dtype=np.uint8).reshape(
            height, width
        )
        raw_depth = np.frombuffer(depth_texture.read(alignment=1), dtype=np.float32).reshape(
            height, width
        )
    finally:
        vertex_array.release()
        buffer.release()
    mask = np.flipud(raw_mask) > 0
    normalized_depth = np.flipud(raw_depth).astype(np.float64)
    metric_depth = np.full(image_shape, np.inf, dtype=np.float64)
    metric_depth[mask] = near / np.maximum(1.0 - normalized_depth[mask], 1e-12)
    return MeshRenderResult(
        mask=mask,
        depth=metric_depth,
        link_index=np.full(image_shape, -1, dtype=np.int32),
        visual_count=len(meshes),
        renderer="opengl_triangle_zbuffer_v1",
        triangle_count=triangle_count,
        rendered_links=tuple(mesh.link for mesh in meshes),
    )


def render_urdf_robot_silhouette(
    *,
    urdf_path: str | Path,
    root_link: str,
    joint_state: dict[str, float],
    camera_from_root: np.ndarray,
    intrinsic: np.ndarray,
    image_shape: tuple[int, int],
    explicit_unrecorded_joint_positions: dict[str, float] | None = None,
    distortion_coefficients: np.ndarray | None = None,
    display_transform: np.ndarray | None = None,
    renderer: Literal["cpu_triangle_zbuffer", "opengl_zbuffer"] = "cpu_triangle_zbuffer",
    included_links: set[str] | None = None,
) -> MeshRenderResult:
    """Render every supported URDF visual at the exact recorded joint state."""

    path = Path(urdf_path)
    root = ET.parse(path).getroot()
    root_from_links = _joint_link_transforms(
        root,
        root_link,
        joint_state,
        explicit_unrecorded_joint_positions=explicit_unrecorded_joint_positions,
    )
    meshes: list[TriangleMesh] = []
    camera_from_visuals: list[np.ndarray] = []
    camera_from_root = np.asarray(camera_from_root, dtype=np.float64).reshape(4, 4)
    for link_node in root.findall("link"):
        link = link_node.attrib["name"]
        if link not in root_from_links:
            continue
        if included_links is not None and link not in included_links:
            continue
        for visual in link_node.findall("visual"):
            geometry = visual.find("geometry")
            if geometry is None:
                continue
            origin = visual.find("origin")
            xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            meshes.append(_visual_mesh(geometry, path, link))
            camera_from_visuals.append(
                camera_from_root @ root_from_links[link] @ _transform_from_xyz_rpy(xyz, rpy)
            )
    if not meshes:
        raise ValueError("URDF contains no reachable supported visual geometry")
    if renderer == "opengl_zbuffer":
        return _rasterize_opengl(
            meshes,
            camera_from_visuals,
            intrinsic,
            image_shape,
            distortion_coefficients=distortion_coefficients,
            display_transform=display_transform,
        )
    return _rasterize(
        meshes,
        camera_from_visuals,
        intrinsic,
        image_shape,
        distortion_coefficients=distortion_coefficients,
        display_transform=display_transform,
    )


def render_profiled_gr3_silhouette(
    *,
    state: Sequence[float],
    state_names: Sequence[str],
    state_units: Sequence[str],
    camera_from_root: np.ndarray,
    kinematics: KinematicsProfile,
    reprojection: ReprojectionProfile,
    renderer: Literal["cpu_triangle_zbuffer", "opengl_zbuffer"] = "cpu_triangle_zbuffer",
    included_links: set[str] | None = None,
) -> MeshRenderResult:
    """Fail closed unless the frozen URDF/name/unit/Camera0 profile is complete."""

    path, error = _urdf_path(kinematics)
    if path is None:
        raise ValueError(error or "trusted GR3 URDF is unavailable")
    if not reprojection.binding_verified or reprojection.stored_intrinsic is None:
        raise ValueError("trusted Camera0 stored-image calibration is unavailable")
    if reprojection.stored_image_size is None:
        raise ValueError("stored-image resolution is unavailable")
    if len(state) != len(state_names) or len(state) != len(state_units):
        raise ValueError("state values/names/units have different dimensions")
    if not kinematics.joint_names or len(kinematics.joint_names) != len(
        kinematics.joint_state_indices
    ):
        raise ValueError("trusted GR3 name/index mapping is incomplete")
    joint_types = {
        joint.attrib.get("name", ""): joint.attrib.get("type", "fixed")
        for joint in ET.parse(path).getroot().findall("joint")
    }
    joint_state: dict[str, float] = {}
    for name, source_index in zip(
        kinematics.joint_names, kinematics.joint_state_indices, strict=True
    ):
        if source_index >= len(state_names) or state_names[source_index] != name:
            raise ValueError(f"recorded state mapping mismatch for joint {name!r}")
        expected_unit = "m" if joint_types.get(name) == "prismatic" else "rad"
        if state_units[source_index] != expected_unit:
            raise ValueError(
                f"recorded unit mismatch for joint {name!r}: expected {expected_unit}"
            )
        value = float(state[source_index])
        if not math.isfinite(value):
            raise ValueError(f"recorded joint {name!r} is non-finite")
        joint_state[name] = value
    if reprojection.stored_video_undistorted is False:
        if (
            reprojection.raw_intrinsic is None
            or reprojection.distortion_model != "opencv_14d"
            or reprojection.distortion_coefficients is None
            or len(reprojection.distortion_coefficients) != 14
            or reprojection.display_transform is None
        ):
            raise ValueError(
                "distorted Camera0 video requires raw K, OpenCV 14D coefficients, and display transform"
            )
        raw_intrinsic = np.asarray(reprojection.raw_intrinsic, dtype=np.float64)
        display_transform = np.asarray(reprojection.display_transform, dtype=np.float64)
        stored_intrinsic = np.asarray(reprojection.stored_intrinsic, dtype=np.float64)
        if not np.allclose(
            display_transform @ raw_intrinsic,
            stored_intrinsic,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("Camera0 raw-K/display transform does not reproduce stored K")
        render_intrinsic = raw_intrinsic
        distortion = np.asarray(reprojection.distortion_coefficients, dtype=np.float64)
    elif reprojection.stored_video_undistorted is True:
        render_intrinsic = np.asarray(reprojection.stored_intrinsic, dtype=np.float64)
        distortion = None
        display_transform = None
    else:
        raise ValueError("Camera0 stored_video_undistorted authority is unspecified")
    width, height = reprojection.stored_image_size
    return render_urdf_robot_silhouette(
        urdf_path=path,
        root_link=kinematics.root_link,
        joint_state=joint_state,
        camera_from_root=camera_from_root,
        intrinsic=render_intrinsic,
        image_shape=(height, width),
        explicit_unrecorded_joint_positions=kinematics.explicit_unrecorded_joint_positions,
        distortion_coefficients=distortion,
        display_transform=display_transform,
        renderer=renderer,
        included_links=included_links,
    )


def _mask_metrics(projected: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    projected = np.asarray(projected, dtype=bool)
    observed = np.asarray(observed, dtype=bool)
    if projected.shape != observed.shape:
        raise ValueError("projected and observed masks must have the same shape")
    intersection = int(np.count_nonzero(projected & observed))
    projected_count = int(np.count_nonzero(projected))
    observed_count = int(np.count_nonzero(observed))
    union = projected_count + observed_count - intersection
    return {
        "iou": intersection / union if union else 0.0,
        "precision": intersection / projected_count if projected_count else 0.0,
        "recall": intersection / observed_count if observed_count else 0.0,
        "coverage": union / projected.size if projected.size else 0.0,
    }


def evaluate_full_mesh_c2_keyframes(
    *,
    projected_masks: Sequence[np.ndarray],
    temporal_projected_masks: Sequence[dict[int, np.ndarray]],
    sam_results: Sequence[Sam3Result],
    frame_indexes: Sequence[int],
    calibration: C2Calibration,
) -> MetricEvidence:
    count = len(projected_masks)
    if not (count == len(temporal_projected_masks) == len(sam_results) == len(frame_indexes)):
        raise ValueError("C2 full-mesh keyframe evidence lengths differ")
    frames = []
    evaluated = []
    for projected, temporal, sam, frame_index in zip(
        projected_masks, temporal_projected_masks, sam_results, frame_indexes, strict=True
    ):
        if sam.status != "evaluated" or sam.mask is None or "mock" in sam.reason.casefold():
            frames.append({"frame_index": int(frame_index), "status": "not_evaluated", "reason": sam.reason})
            continue
        metrics = _mask_metrics(projected, sam.mask)
        offsets = {
            int(offset): _mask_metrics(mask, sam.mask)["iou"]
            for offset, mask in temporal.items()
        }
        best_offset = max(offsets, key=offsets.get) if offsets else 0
        row = {
            "frame_index": int(frame_index),
            "status": "evaluated",
            **metrics,
            "sam3_confidence": sam.confidence,
            "sam3_cache_hit": sam.cache_hit,
            "sam3_cache_key": sam.cache_key,
            "temporal_offset_diagnostic": {
                "iou_by_offset_frames": offsets,
                "best_offset_frames": int(best_offset),
                "exact_frame_is_best": best_offset == 0,
            },
        }
        frames.append(row)
        evaluated.append(metrics)
    fidelity = "calibrated" if calibration.calibrated else "provisional"
    common = {
        "schema": "qwen-robotmanip-c2-full-silhouette-v2",
        "projection_scope": "full URDF visual silhouette",
        "renderer": "cpu_triangle_zbuffer_v1",
        "provider": "SAM3",
        "mask_checkpoint_kind": calibration.mask_checkpoint_kind,
        "camera_id": calibration.camera_id,
        "embodiment": calibration.embodiment,
        "calibration_version": calibration.version,
        "threshold_version": calibration.threshold_version,
        "hard_reject_allowed": calibration.calibrated,
        "frame_evidence": frames,
    }
    if not evaluated:
        return MetricEvidence(
            metric_id="C2",
            label=QWEN_CORE_LABELS["C2"],
            group="qwen_core",
            status="not_evaluated",
            fidelity=fidelity,
            weight=QWEN_CORE_DEFAULT_WEIGHTS["C2"],
            reason="no frame has both full-mesh projection and non-mock SAM3 mask",
            evidence=common,
        )
    aggregate = {
        name: float(np.mean([row[name] for row in evaluated]))
        for name in ("iou", "precision", "recall", "coverage")
    }
    return MetricEvidence(
        metric_id="C2",
        label=QWEN_CORE_LABELS["C2"],
        group="qwen_core",
        status="evaluated",
        score=aggregate["iou"],
        fidelity=fidelity,
        weight=QWEN_CORE_DEFAULT_WEIGHTS["C2"],
        reason=(
            "full-URDF-visual / observed robot mask evidence; review-only until per-camera calibration"
            if not calibration.calibrated
            else "calibrated full-URDF-visual / observed robot mask evidence"
        ),
        evidence={
            **common,
            "evaluated_frame_count": len(evaluated),
            "clip_aggregation": aggregate,
            "review_threshold": calibration.iou_review_threshold,
            "review_recommended": aggregate["iou"] < calibration.iou_review_threshold,
        },
    )
