"""Rectangular counterpart to IsaacLab's square-only random grid terrain."""
import numpy as np
import torch
import trimesh


def random_grid_terrain(difficulty, cfg):
    size_x, size_y = cfg.size
    grid_width = cfg.grid_width
    num_x, num_y = int(size_x / grid_width), int(size_y / grid_width)
    if num_x < 1 or num_y < 1:
        raise ValueError(f"Grid width {grid_width} does not fit terrain size {cfg.size}.")

    grid_height = cfg.grid_height_range[0] + difficulty * (cfg.grid_height_range[1] - cfg.grid_height_range[0])
    terrain_height = 1.0
    inner_x, inner_y = num_x * grid_width, num_y * grid_width
    border_x, border_y = max(size_x - inner_x, 0.0), max(size_y - inner_y, 0.0)
    meshes = []

    def border(dimensions, position):
        meshes.append(trimesh.creation.box(dimensions, trimesh.transformations.translation_matrix(position)))

    if border_x > 1.0e-8:
        border((border_x / 2, inner_y, terrain_height), (border_x / 4, size_y / 2, -terrain_height / 2))
        border((border_x / 2, inner_y, terrain_height),
               (size_x - border_x / 4, size_y / 2, -terrain_height / 2))
    if border_y > 1.0e-8:
        border((size_x, border_y / 2, terrain_height), (size_x / 2, border_y / 4, -terrain_height / 2))
        border((size_x, border_y / 2, terrain_height),
               (size_x / 2, size_y - border_y / 4, -terrain_height / 2))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    template = trimesh.creation.box(
        (grid_width, grid_width, terrain_height),
        trimesh.transformations.translation_matrix((grid_width / 2, grid_width / 2, -terrain_height / 2)),
    )
    vertices = torch.tensor(template.vertices, device=device).repeat(num_x * num_y, 1, 1)
    x, y = torch.meshgrid(torch.arange(num_x, device=device), torch.arange(num_y, device=device), indexing="ij")
    offsets = torch.stack((x.flatten(), y.flatten()), dim=1) * grid_width
    offsets += offsets.new_tensor((border_x / 2, border_y / 2))
    vertices[:, :, :2] += offsets.unsqueeze(1)

    if cfg.holes:
        keep_x = torch.logical_and(
            (vertices[:, :, 0] > (size_x - border_x - cfg.platform_width) / 2).all(dim=1),
            (vertices[:, :, 0] < (size_x + border_x + cfg.platform_width) / 2).all(dim=1),
        )
        keep_y = torch.logical_and(
            (vertices[:, :, 1] > (size_y - border_y - cfg.platform_width) / 2).all(dim=1),
            (vertices[:, :, 1] < (size_y + border_y + cfg.platform_width) / 2).all(dim=1),
        )
        vertices = torch.cat((vertices[keep_x], vertices[keep_y]))

    count = len(vertices)
    heights = torch.empty(count, device=device).uniform_(-grid_height, grid_height)
    top_vertices = vertices[:, :, 2] == 0
    vertices[:, :, 2] += top_vertices * heights[:, None]
    faces = torch.tensor(template.faces, device=device).repeat(count, 1, 1)
    faces += torch.arange(count, device=device).view(-1, 1, 1) * len(template.vertices)
    grid_mesh = trimesh.Trimesh(vertices=vertices.reshape(-1, 3).cpu().numpy(),
                                faces=faces.reshape(-1, 3).cpu().numpy())
    meshes.append(grid_mesh)

    platform = trimesh.creation.box(
        (cfg.platform_width, cfg.platform_width, terrain_height + grid_height),
        trimesh.transformations.translation_matrix((size_x / 2, size_y / 2, -terrain_height / 2 + grid_height / 2)),
    )
    meshes.append(platform)
    return meshes, np.array((size_x / 2, size_y / 2, grid_height))
