package main

import mikk "../"

import "core:fmt"

//odinfmt: disable
cube_indices := [36]u32 {
	2,  5,  11, 2, 
    11, 8,  6,  9,
    21, 6,  21, 18,
    20, 23, 17, 20,
    17, 14, 12, 15,
    3,  12, 3,  0,
	7,  19, 13, 7,
    13, 1,  22, 10, 
    4,  22, 4,  1,
}

cube_positions := [24][3]f32 {
    {-1, -1,  1}, {-1, -1,  1}, {-1, -1,  1}, {-1,  1,  1},
    {-1,  1,  1}, {-1,  1,  1}, {-1, -1, -1}, {-1, -1, -1},
    {-1, -1, -1}, {-1,  1, -1}, {-1,  1, -1}, {-1,  1, -1},
    { 1, -1,  1}, { 1, -1,  1}, { 1, -1,  1}, { 1,  1,  1},
    { 1,  1,  1}, { 1,  1,  1}, { 1, -1, -1}, { 1, -1, -1},
    { 1, -1, -1}, { 1,  1, -1}, { 1,  1, -1}, { 1,  1, -1},
}

cube_normals := [24][3]f32 {
    { 0,  0,  1}, { 0, -1,  0}, {-1,  0,  0}, { 0,  0,  1},
    { 0,  1,  0}, {-1,  0,  0}, { 0,  0, -1}, { 0, -1,  0},
    {-1,  0,  0}, { 0,  0, -1}, { 0,  1,  0}, {-1,  0,  0},
    { 0,  0,  1}, { 0, -1,  0}, { 1,  0,  0}, { 0,  0,  1},
    { 0,  1,  0}, { 1,  0,  0}, { 0,  0, -1}, { 0, -1,  0},
    { 1,  0,  0}, { 0,  0, -1}, { 0,  1,  0}, { 1,  0,  0},
}

cube_uvs := [24][2]f32 {
    { 0, 1 }, { 0, 1 }, { 0, 1 }, { 0, 0 },
    { 0, 1 }, { 0, 0 }, { 0, 1 }, { 0, 0 },
    { 1, 1 }, { 0, 0 }, { 0, 0 }, { 1, 0 },
    { 1, 1 }, { 1, 1 }, { 0, 1 }, { 1, 0 },
    { 1, 1 }, { 0, 0 }, { 1, 1 }, { 1, 0 },
    { 1, 1 }, { 1, 0 }, { 1, 0 }, { 1, 0 },
}

cube_check_tangents := [24][4]f32 {
    {-1, 0, 0, -1}, { 1, 0, 0, -1}, { 0, 0, 1,  1}, {-1, 0, 0, -1},
    {-1, 0, 0, -1}, { 0, 0, 1,  1}, {-1, 0, 0,  1}, {-1, 0, 0,  1},
    { 0, 0, 1,  1}, {-1, 0, 0,  1}, {-1, 0, 0, -1}, { 0, 0, 1,  1},
    {-1, 0, 0, -1}, {-1, 0, 0,  1}, { 0, 0, 1, -1}, {-1, 0, 0, -1},
    { 0, 0, 0,  0}, { 0, 0, 1, -1}, {-1, 0, 0,  1}, {-1, 0, 0,  1},
    { 0, 0, 1, -1}, {-1, 0, 0,  1}, {-1, 0, 0, -1}, { 0, 0, 1, -1},
}
//odinfmt: enable

Mesh :: struct {
	indices:   []u32,
	positions: [][3]f32,
	normals:   [][3]f32,
	uvs:       [][2]f32,
	tangents:  [][4]f32,
}

get_vertex_index :: proc(pContext: ^mikk.Context, iFace: int, iVert: int) -> int {
	mesh := cast(^Mesh)pContext.user_data

	indices_index := iVert + (iFace * get_num_vertices_of_face(pContext, iFace))
	index := mesh.indices[indices_index]

	return int(index)
}

get_num_faces :: proc(pContext: ^mikk.Context) -> int {
	mesh := cast(^Mesh)pContext.user_data
	return len(mesh.indices) / 3
}

get_num_vertices_of_face :: proc(pContext: ^mikk.Context, iFace: int) -> int {
	return 3
}

get_position :: proc(pContext: ^mikk.Context, iFace: int, iVert: int) -> [3]f32 {
	mesh := cast(^Mesh)pContext.user_data
	return mesh.positions[get_vertex_index(pContext, iFace, iVert)]
}

get_normal :: proc(pContext: ^mikk.Context, iFace: int, iVert: int) -> [3]f32 {
	mesh := cast(^Mesh)pContext.user_data
	return mesh.normals[get_vertex_index(pContext, iFace, iVert)]
}

get_tex_coord :: proc(pContext: ^mikk.Context, iFace: int, iVert: int) -> [2]f32 {
	mesh := cast(^Mesh)pContext.user_data
	return mesh.uvs[get_vertex_index(pContext, iFace, iVert)]
}

set_t_space_basic :: proc(pContext: ^mikk.Context, fvTangent: [3]f32, fSign: f32, iFace: int, iVert: int) {
	mesh := cast(^Mesh)pContext.user_data
	index := get_vertex_index(pContext, iFace, iVert)

	mesh.tangents[index].xyz = fvTangent
	mesh.tangents[index].w = fSign
}

main :: proc() {
	gen_tangents: [24][4]f32

	mesh_data := Mesh {
		indices   = cube_indices[:],
		positions = cube_positions[:],
		normals   = cube_normals[:],
		uvs       = cube_uvs[:],
		tangents  = gen_tangents[:],
	}

	interface := mikk.Interface {
		get_num_faces            = get_num_faces,
		get_num_vertices_of_face = get_num_vertices_of_face,
		get_position             = get_position,
		get_normal               = get_normal,
		get_tex_coord            = get_tex_coord,
		set_t_space_basic        = set_t_space_basic,
		//set_t_space              = set_t_space, // for more uses beyond basic normal-mapping.
	}

	ctx := mikk.Context {
		interface = &interface,
		user_data = &mesh_data,
	}

	ok := mikk.generate_tangents(&ctx)
	assert(ok)

	for i in 0 ..< len(gen_tangents) {
		assert(gen_tangents[i] == cube_check_tangents[i])
	}

	fmt.println("Tangents generated!")
	fmt.println(gen_tangents)
}
