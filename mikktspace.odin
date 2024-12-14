/**
 *  Port of Morten S. Mikkelsen's original tangent space algorithm
 *  implementation written in C, with some style changes to fit idiomatic
 *  Odin style. Original source: https://github.com/mmikk/MikkTSpace
 *
 *  Original work: Copyright (C) 2011 by Morten S. Mikkelsen
 *  Modified work: Copyright (C) 2024 by Matthew Taylor
 *
 *  This software is provided 'as-is', without any express or implied
 *  warranty.  In no event will the authors be held liable for any damages
 *  arising from the use of this software.
 *
 *  Permission is granted to anyone to use this software for any purpose,
 *  including commercial applications, and to alter it and redistribute it
 *  freely, subject to the following restrictions:
 *
 *  1. The origin of this software must not be misrepresented; you must not
 *     claim that you wrote the original software. If you use this software
 *     in a product, an acknowledgment in the product documentation would be
 *     appreciated but is not required.
 *  2. Altered source versions must be plainly marked as such, and must not be
 *     misrepresented as being the original software.
 *  3. This notice may not be removed or altered from any source distribution.
 */

package mikktspace

import "core:math"
import "core:math/linalg"

Interface :: struct {
	// Returns the number of faces (triangles/quads) on the mesh to be processed.
	get_num_faces:            proc(pContext: ^Context) -> int,

	// Returns the number of vertices on face number iFace
	// iFace is a number in the range {0, 1, ..., getNumFaces()-1}
	get_num_vertices_of_face: proc(pContext: ^Context, iFace: int) -> int,

	// returns the position/normal/texcoord of the referenced face of vertex number iVert.
	// iVert is in the range {0,1,2} for triangles and {0,1,2,3} for quads.
	get_position:             proc(pContext: ^Context, iFace: int, iVert: int) -> [3]f32,
	get_normal:               proc(pContext: ^Context, iFace: int, iVert: int) -> [3]f32,
	get_tex_coord:            proc(pContext: ^Context, iFace: int, iVert: int) -> [2]f32,

	// either (or both) of the two setTSpace callbacks can be set.
	// The call-back m_setTSpaceBasic() is sufficient for basic normal mapping.

	// This function is used to return the tangent and fSign to the application.
	// fvTangent is a unit length vector.
	// For normal maps it is sufficient to use the following simplified version of the bitangent which is generated at pixel/vertex level.
	// bitangent = fSign * cross(vN, tangent);
	// Note that the results are returned unindexed. It is possible to generate a new index list
	// But averaging/overwriting tangent spaces by using an already existing index list WILL produce INCORRECT results.
	// DO NOT! use an already existing index list.
	set_t_space_basic:        proc(pContext: ^Context, fvTangent: [3]f32, fSign: f32, iFace: int, iVert: int),

	// This function is used to return tangent space results to the application.
	// fvTangent and fvBiTangent are unit length vectors and fMagS and fMagT are their
	// true magnitudes which can be used for relief mapping effects.
	// fvBiTangent is the "real" bitangent and thus may not be perpendicular to fvTangent.
	// However, both are perpendicular to the vertex normal.
	// For normal maps it is sufficient to use the following simplified version of the bitangent which is generated at pixel/vertex level.
	// fSign = bIsOrientationPreserving ? 1.0f : (-1.0f);
	// bitangent = fSign * cross(vN, tangent);
	// Note that the results are returned unindexed. It is possible to generate a new index list
	// But averaging/overwriting tangent spaces by using an already existing index list WILL produce INCRORRECT results.
	// DO NOT! use an already existing index list.
	set_t_space:              proc(
		pContext: ^Context,
		fvTangent: [3]f32,
		fvBiTangent: [3]f32,
		fMagS: f32,
		fMagT: f32,
		bIsOrientationPreserving: bool,
		iFace: int,
		iVert: int,
	),
}

Context :: struct {
	interface: ^Interface, // initialized with callback functions
	user_data: rawptr, // pointer to client side mesh data etc. (passed as the first parameter with every interface call)
}

// Generates tangents using the provided interface and user data.
// Returns true if the operation succeeded, otherwise false.
generate_tangents :: proc(pContext: ^Context, fAngularThreshold: f32 = 180.0, allocator := context.allocator) -> bool {
	// count nr_triangles
	iNrTrianglesIn: int
	iNrTSPaces, iTotTris, iDegenTriangles, iNrMaxGroups: int
	iNrActiveGroups, index: int
	iNrFaces := pContext.interface.get_num_faces(pContext)
	bRes: bool
	fThresCos := math.cos((fAngularThreshold * f32(math.PI)) / 180.0)

	// verify all call-backs have been set
	if pContext.interface.get_num_faces == nil ||
	   pContext.interface.get_num_vertices_of_face == nil ||
	   pContext.interface.get_position == nil ||
	   pContext.interface.get_normal == nil ||
	   pContext.interface.get_tex_coord == nil {
		return false
	}

	// count triangles on supported faces
	for f in 0 ..< iNrFaces {
		verts := pContext.interface.get_num_vertices_of_face(pContext, f)
		if verts == 3 do iNrTrianglesIn += 1
		else if verts == 4 do iNrTrianglesIn += 2
	}
	if iNrTrianglesIn <= 0 do return false

	// allocate memory for an index list
	piTriListIn := make([]int, 3 * iNrTrianglesIn, allocator)
	if piTriListIn == nil {
		return false
	}
	defer delete(piTriListIn)

	pTriInfos := make([]Tri_Info, iNrTrianglesIn, allocator)
	if pTriInfos == nil {
		return false
	}
	defer delete(pTriInfos)

	// make an initial triangle . face index list
	iNrTSPaces = generate_initial_vertices_index_list(pTriInfos, piTriListIn, pContext, iNrTrianglesIn)

	// make a welded index list of identical positions and attributes (pos, norm, texc)
	generate_shared_vertices_index_list(piTriListIn, pContext, iNrTrianglesIn, allocator)

	// Mark all degenerate triangles
	iTotTris = iNrTrianglesIn
	iDegenTriangles = 0
	for t in 0 ..< iTotTris {
		i0 := piTriListIn[t * 3 + 0]
		i1 := piTriListIn[t * 3 + 1]
		i2 := piTriListIn[t * 3 + 2]
		p0 := get_position(pContext, i0)
		p1 := get_position(pContext, i1)
		p2 := get_position(pContext, i2)
		if (p0 == p1) || (p0 == p2) || (p1 == p2) { 	// degenerate
			pTriInfos[t].iFlag |= {.MarkDegenerate}
			iDegenTriangles += 1
		}
	}
	iNrTrianglesIn = iTotTris - iDegenTriangles

	// mark all triangle pairs that belong to a quad with only one
	// good triangle. These need special treatment in DegenEpilogue().
	// Additionally, move all good triangles to the start of
	// pTriInfos[] and piTriListIn[] without changing order and
	// put the degenerate triangles last.
	degen_prologue(pTriInfos, piTriListIn, iNrTrianglesIn, iTotTris)

	// evaluate triangle level attributes and neighbor list
	init_tri_info(pTriInfos, piTriListIn, pContext, iNrTrianglesIn, allocator)

	// based on the 4 rules, identify groups based on connectivity
	iNrMaxGroups = iNrTrianglesIn * 3

	pGroups := make([]Group, iNrMaxGroups, allocator)
	defer delete(pGroups)

	piGroupTrianglesBuffer := make([]int, iNrTrianglesIn * 3, allocator)
	defer delete(piGroupTrianglesBuffer)

	if pGroups == nil || piGroupTrianglesBuffer == nil {
		return false
	}

	iNrActiveGroups = build_4_rule_groups(pTriInfos, pGroups, piGroupTrianglesBuffer, piTriListIn, iNrTrianglesIn)

	psTspace := make([]T_Space, iNrTSPaces, allocator)
	defer delete(psTspace)

	if psTspace == nil {
		return false
	}

	for t in 0 ..< iNrTSPaces {
		psTspace[t].vOs.x = 1.0
		psTspace[t].vOs.y = 0.0
		psTspace[t].vOs.z = 0.0
		psTspace[t].fMagS = 1.0
		psTspace[t].vOt.x = 0.0
		psTspace[t].vOt.y = 1.0
		psTspace[t].vOt.z = 0.0
		psTspace[t].fMagT = 1.0
	}

	// make tspaces, each group is split up into subgroups if necessary
	// based on fAngularThreshold. Finally a tangent space is made for
	// every resulting subgroup
	bRes = generate_t_spaces(psTspace, pTriInfos, pGroups, iNrActiveGroups, piTriListIn, fThresCos, pContext, allocator)

	// clean up

	if !bRes {
		return false
	}


	// degenerate quads with one good triangle will be fixed by copying a space from
	// the good triangle to the coinciding vertex.
	// all other degenerate triangles will just copy a space from any good triangle
	// with the same welded index in piTriListIn[].
	degen_epilogue(psTspace, pTriInfos, piTriListIn, pContext, iNrTrianglesIn, iTotTris)

	index = 0
	for f in 0 ..< iNrFaces {
		verts := pContext.interface.get_num_vertices_of_face(pContext, f)
		if verts != 3 && verts != 4 do continue


		// I've decided to let degenerate triangles and group-with-anythings
		// vary between left/right hand coordinate systems at the vertices.
		// All healthy triangles on the other hand are built to always be either or.

		/*// force the coordinate system orientation to be uniform for every face.
		// (this is already the case for good triangles but not for
		// degenerate ones and those with bGroupWithAnything==true)
		bool bOrient = psTspace[index].bOrient;
		if psTspace[index].iCounter == 0 do	// tspace was not derived from a group
		{
			// look for a space created in GenerateTSpaces() by iCounter>0
			bool bNotFound = true;
			int i=1;
			for (i<verts && bNotFound)
			{
				if psTspace[index+i].iCounter > 0 do bNotFound=false;
				else ++i;
			}
			if !bNotFound do bOrient = psTspace[index+i].bOrient;
		}*/

		// set data
		for i in 0 ..< verts {
			pTSpace: ^T_Space = &psTspace[index]
			tang: [3]f32 = {pTSpace.vOs.x, pTSpace.vOs.y, pTSpace.vOs.z}
			bitang: [3]f32 = {pTSpace.vOt.x, pTSpace.vOt.y, pTSpace.vOt.z}

			if pContext.interface.set_t_space != nil {
				pContext.interface.set_t_space(pContext, tang, bitang, pTSpace.fMagS, pTSpace.fMagT, pTSpace.bOrient, f, i)
			}

			if pContext.interface.set_t_space_basic != nil {
				pContext.interface.set_t_space_basic(pContext, tang, pTSpace.bOrient == true ? 1.0 : (-1.0), f, i)
			}

			index += 1
		}
	}

	return true
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

@(private)
INTERNAL_RND_SORT_SEED: u32 : 39871946

@(private)
Vec3 :: [3]f32

@(private)
Sub_Group :: struct {
	iNrFaces:    int,
	pTriMembers: []int,
}

@(private)
Group :: struct {
	iNrFaces:              int,
	pFaceIndices:          []int,
	iVertexRepresentitive: int,
	bOrientPreservering:   bool,
}

@(private)
Tri_Flag :: enum {
	MarkDegenerate,
	QuadOneDegenTri,
	GroupWithAny,
	OrientPreserving,
}

@(private)
Tri_Flags :: bit_set[Tri_Flag]

@(private)
Tri_Info :: struct {
	FaceNeighbors:  [3]int,
	AssignedGroup:  [3]^Group,

	// normalized first order face derivatives
	vOs, vOt:       Vec3,
	fMagS, fMagT:   f32,

	// determines if the current and the next triangle are a quad.
	iOrgFaceNumber: int,
	iFlag:          Tri_Flags,
	iTSpacesOffs:   int,
	vert_num:       [4]u8,
}

@(private)
T_Space :: struct {
	vOs:      Vec3,
	fMagS:    f32,
	vOt:      Vec3,
	fMagT:    f32,
	iCounter: int, // this is to average back into quads.
	bOrient:  bool,
}

@(private)
make_index :: proc(iFace: int, iVert: int) -> int {
	assert(iVert >= 0 && iVert < 4 && iFace >= 0)
	return (iFace << 2) | (iVert & 0x3)
}

@(private)
index_to_data :: proc(piFace: ^int, piVert: ^int, iIndexIn: int) {
	piVert^ = iIndexIn & 0x3
	piFace^ = iIndexIn >> 2
}

@(private)
avg_t_space :: proc(pTS0: ^T_Space, pTS1: ^T_Space) -> T_Space {
	ts_res: T_Space

	// this if is important. Due to floating point precision
	// averaging when ts0==ts1 will cause a slight difference
	// which results in tangent space splits later on
	if pTS0.fMagS == pTS1.fMagS && pTS0.fMagT == pTS1.fMagT && pTS0.vOs == pTS1.vOs && pTS0.vOt == pTS1.vOt {
		ts_res.fMagS = pTS0.fMagS
		ts_res.fMagT = pTS0.fMagT
		ts_res.vOs = pTS0.vOs
		ts_res.vOt = pTS0.vOt
	} else {
		ts_res.fMagS = 0.5 * (pTS0.fMagS + pTS1.fMagS)
		ts_res.fMagT = 0.5 * (pTS0.fMagT + pTS1.fMagT)
		ts_res.vOs = pTS0.vOs + pTS1.vOs
		ts_res.vOt = pTS0.vOt + pTS1.vOt
		if ts_res.vOs != 0 do ts_res.vOs = linalg.normalize(ts_res.vOs)
		if ts_res.vOt != 0 do ts_res.vOt = linalg.normalize(ts_res.vOt)
	}

	return ts_res
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

@(private)
Tmp_Vert :: struct {
	vert:  [3]f32,
	index: int,
}

@(private)
G_CELLS :: 2048

// it is IMPORTANT that this function is called to evaluate the hash since
// inlining could potentially reorder instructions and generate different
// results for the same effective input value fVal.
// FindGridCell
@(private)
find_grid_cell :: #force_no_inline proc(fMin: f32, fMax: f32, fVal: f32) -> int {
	fIndex := f32(G_CELLS) * ((fVal - fMin) / (fMax - fMin))
	iIndex := int(fIndex)
	return iIndex < G_CELLS ? (iIndex >= 0 ? iIndex : 0) : (G_CELLS - 1)
}

@(private)
generate_shared_vertices_index_list :: proc(piTriList_in_and_out: []int, pContext: ^Context, iNrTrianglesIn: int, allocator := context.allocator) {
	// Generate bounding box
	iChannel: int
	iMaxCount: int
	vMin := get_position(pContext, 0)
	vMax := vMin
	vDim: Vec3
	fMin, fMax: f32
	for i in 1 ..< (iNrTrianglesIn * 3) {
		index := piTriList_in_and_out[i]

		vP := get_position(pContext, index)
		if vMin.x > vP.x do vMin.x = vP.x
		else if vMax.x < vP.x do vMax.x = vP.x
		if vMin.y > vP.y do vMin.y = vP.y
		else if vMax.y < vP.y do vMax.y = vP.y
		if vMin.z > vP.z do vMin.z = vP.z
		else if vMax.z < vP.z do vMax.z = vP.z
	}

	vDim = vMax - vMin
	iChannel = 0
	fMin = vMin.x
	fMax = vMax.x
	if vDim.y > vDim.x && vDim.y > vDim.z {
		iChannel = 1
		fMin = vMin.y
		fMax = vMax.y
	} else if vDim.z > vDim.x {
		iChannel = 2
		fMin = vMin.z
		fMax = vMax.z
	}

	// make allocations
	piHashTable := make([]int, iNrTrianglesIn * 3, allocator)
	defer delete(piHashTable)

	piHashCount := make([]int, G_CELLS, allocator)
	defer delete(piHashCount)

	piHashOffsets := make([]int, G_CELLS, allocator)
	defer delete(piHashOffsets)

	piHashCount2 := make([]int, G_CELLS, allocator)
	defer delete(piHashCount2)

	if piHashTable == nil || piHashCount == nil || piHashOffsets == nil || piHashCount2 == nil {
		generate_shared_vertices_index_list_slow(piTriList_in_and_out, pContext, iNrTrianglesIn)
		return
	}

	// count amount of elements in each cell unit
	for i in 0 ..< (iNrTrianglesIn * 3) {
		index := piTriList_in_and_out[i]
		vP := get_position(pContext, index)
		fVal := iChannel == 0 ? vP.x : (iChannel == 1 ? vP.y : vP.z)
		iCell := find_grid_cell(fMin, fMax, fVal)
		piHashCount[iCell] += 1
	}

	// evaluate start index of each cell.
	piHashOffsets[0] = 0
	for k in 1 ..< G_CELLS {
		piHashOffsets[k] = piHashOffsets[k - 1] + piHashCount[k - 1]
	}

	// insert vertices
	for i in 0 ..< (iNrTrianglesIn * 3) {
		index := piTriList_in_and_out[i]
		vP := get_position(pContext, index)
		fVal := iChannel == 0 ? vP.x : (iChannel == 1 ? vP.y : vP.z)
		iCell := find_grid_cell(fMin, fMax, fVal)

		assert(piHashCount2[iCell] < piHashCount[iCell])
		pTable := piHashTable[piHashOffsets[iCell]:]
		pTable[piHashCount2[iCell]] = i // vertex i has been inserted.
		piHashCount2[iCell] += 1
	}

	for k in 0 ..< G_CELLS {
		assert(piHashCount2[k] == piHashCount[k]) // verify the count
	}

	// find maximum amount of entries in any hash entry
	iMaxCount = piHashCount[0]

	for k in 1 ..< G_CELLS {
		if iMaxCount < piHashCount[k] {
			iMaxCount = piHashCount[k]
		}
	}

	pTmpVert := make([]Tmp_Vert, iMaxCount, allocator)
	defer delete(pTmpVert)

	// complete the merge
	for k in 0 ..< G_CELLS {
		// extract table of cell k and amount of entries in it
		pTable := piHashTable[piHashOffsets[k]:]
		iEntries := piHashCount[k]
		if iEntries < 2 do continue

		if pTmpVert != nil {
			for e in 0 ..< iEntries {
				i := pTable[e]
				vP := get_position(pContext, piTriList_in_and_out[i])
				pTmpVert[e].vert[0] = vP.x
				pTmpVert[e].vert[1] = vP.y
				pTmpVert[e].vert[2] = vP.z
				pTmpVert[e].index = i
			}
			merge_verts_fast(piTriList_in_and_out, pTmpVert, pContext, 0, iEntries - 1)
		} else {
			merge_verts_slow(piTriList_in_and_out, pContext, pTable, iEntries)
		}
	}
}

@(private)
merge_verts_fast :: proc(piTriList_in_and_out: []int, pTmpVert: []Tmp_Vert, pContext: ^Context, iL_in: int, iR_in: int) {
	// make bbox
	fvMin, fvMax: [3]f32
	dx, dy, dz, fSep: f32
	for c in 0 ..< 3 {
		fvMin[c] = pTmpVert[iL_in].vert[c]
		fvMax[c] = fvMin[c]
	}
	for l in (iL_in + 1) ..= iR_in {
		for c in 0 ..< 3 {
			if fvMin[c] > pTmpVert[l].vert[c] do fvMin[c] = pTmpVert[l].vert[c]
			if fvMax[c] < pTmpVert[l].vert[c] do fvMax[c] = pTmpVert[l].vert[c]
		}
	}

	dx = fvMax[0] - fvMin[0]
	dy = fvMax[1] - fvMin[1]
	dz = fvMax[2] - fvMin[2]

	channel := 0
	if dy > dx && dy > dz do channel = 1
	else if dz > dx do channel = 2

	fSep = 0.5 * (fvMax[channel] + fvMin[channel])

	// stop if all vertices are NaNs
	if math.is_nan(fSep) || math.is_inf(fSep) do return

	// terminate recursion when the separation/average value
	// is no longer strictly between fMin and fMax values.
	if fSep >= fvMax[channel] || fSep <= fvMin[channel] {
		// complete the weld
		for l in 0 ..= iR_in {
			i := pTmpVert[l].index
			index := piTriList_in_and_out[i]
			vP := get_position(pContext, index)
			vN := get_normal(pContext, index)
			vT := get_tex_coord(pContext, index)

			bNotFound := true
			l2 := iL_in
			i2rec := -1
			for l2 < l && bNotFound {
				i2 := pTmpVert[l2].index
				index2 := piTriList_in_and_out[i2]
				vP2 := get_position(pContext, index2)
				vN2 := get_normal(pContext, index2)
				vT2 := get_tex_coord(pContext, index2)
				i2rec = i2

				if vP == vP2 && vN == vN2 && vT == vT2 {
					bNotFound = false
				} else {
					l2 += 1
				}
			}

			// merge if previously found
			if !bNotFound {
				piTriList_in_and_out[i] = piTriList_in_and_out[i2rec]
			}
		}
	} else {
		iL := iL_in
		iR := iR_in
		assert((iR_in - iL_in) > 0) // at least 2 entries

		// separate (by fSep) all points between iL_in and iR_in in pTmpVert[]
		for (iL < iR) {
			bReadyLeftSwap := false
			bReadyRightSwap := false
			for ((!bReadyLeftSwap) && iL < iR) {
				assert(iL >= iL_in && iL <= iR_in)
				bReadyLeftSwap = !(pTmpVert[iL].vert[channel] < fSep)
				if !bReadyLeftSwap do iL += 1
			}
			for ((!bReadyRightSwap) && iL < iR) {
				assert(iR >= iL_in && iR <= iR_in)
				bReadyRightSwap = pTmpVert[iR].vert[channel] < fSep
				if !bReadyRightSwap do iR -= 1
			}
			assert((iL < iR) || !(bReadyLeftSwap && bReadyRightSwap))

			if bReadyLeftSwap && bReadyRightSwap {
				sTmp: Tmp_Vert = pTmpVert[iL]
				assert(iL < iR)
				pTmpVert[iL] = pTmpVert[iR]
				pTmpVert[iR] = sTmp
				iL += 1
				iR -= 1
			}
		}

		assert(iL == (iR + 1) || (iL == iR))
		if iL == iR {
			bReadyRightSwap := pTmpVert[iR].vert[channel] < fSep
			if bReadyRightSwap {
				iL += 1
			} else {
				iR -= 1
			}
		}

		// only need to weld when there is more than 1 instance of the (x,y,z)
		if iL_in < iR {
			merge_verts_fast(piTriList_in_and_out, pTmpVert, pContext, iL_in, iR) // weld all left of fSep
		}
		if iL < iR_in {
			merge_verts_fast(piTriList_in_and_out, pTmpVert, pContext, iL, iR_in) // weld all right of (or equal to) fSep
		}
	}
}

@(private)
merge_verts_slow :: proc(piTriList_in_and_out: []int, pContext: ^Context, pTable: []int, iEntries: int) {
	// this can be optimized further using a tree structure or more hashing.
	for e in 0 ..< iEntries {
		i := pTable[e]
		index := piTriList_in_and_out[i]
		vP := get_position(pContext, index)
		vN := get_normal(pContext, index)
		vT := get_tex_coord(pContext, index)

		bNotFound := true
		e2 := 0
		i2rec := -1

		for e2 < e && bNotFound {
			i2 := pTable[e2]
			index2 := piTriList_in_and_out[i2]
			vP2 := get_position(pContext, index2)
			vN2 := get_normal(pContext, index2)
			vT2 := get_tex_coord(pContext, index2)
			i2rec = i2

			if (vP == vP2) && (vN == vN2) && (vT == vT2) {
				bNotFound = false
			} else {
				e2 += 1
			}
		}

		// merge if previously found
		if !bNotFound {
			piTriList_in_and_out[i] = piTriList_in_and_out[i2rec]
		}
	}
}

@(private)
generate_shared_vertices_index_list_slow :: proc(piTriList_in_and_out: []int, pContext: ^Context, iNrTrianglesIn: int) {
	iNumUniqueVerts := 0
	for t in 0 ..< iNrTrianglesIn {
		for i in 0 ..< 3 {
			offs := t * 3 + i
			index := piTriList_in_and_out[offs]

			vP: Vec3 = get_position(pContext, index)
			vN: Vec3 = get_normal(pContext, index)
			vT: Vec3 = get_tex_coord(pContext, index)

			bFound := false
			t2 := 0
			index2rec := -1
			for !bFound && t2 <= t {
				j := 0
				for (!bFound && j < 3) {
					index2 := piTriList_in_and_out[t2 * 3 + j]
					vP2 := get_position(pContext, index2)
					vN2 := get_normal(pContext, index2)
					vT2 := get_tex_coord(pContext, index2)

					if (vP == vP2) && (vN == vN2) && (vT == vT2) {
						bFound = true
					} else {
						j += 1
					}
				}

				if !bFound do t2 += 1
			}

			assert(bFound)
			// if we found our own
			if index2rec == index do iNumUniqueVerts += 1

			piTriList_in_and_out[offs] = index2rec
		}
	}
}

@(private)
generate_initial_vertices_index_list :: proc(pTriInfos: []Tri_Info, piTriList_out: []int, pContext: ^Context, iNrTrianglesIn: int) -> int {
	iTSpacesOffs := 0
	iDstTriIndex := 0

	for f in 0 ..< pContext.interface.get_num_faces(pContext) {
		verts := pContext.interface.get_num_vertices_of_face(pContext, f)
		if verts != 3 && verts != 4 do continue

		pTriInfos[iDstTriIndex].iOrgFaceNumber = f
		pTriInfos[iDstTriIndex].iTSpacesOffs = iTSpacesOffs

		if verts == 3 {
			pVerts := &pTriInfos[iDstTriIndex].vert_num
			pVerts[0] = 0
			pVerts[1] = 1
			pVerts[2] = 2
			piTriList_out[iDstTriIndex * 3 + 0] = make_index(f, 0)
			piTriList_out[iDstTriIndex * 3 + 1] = make_index(f, 1)
			piTriList_out[iDstTriIndex * 3 + 2] = make_index(f, 2)
			iDstTriIndex += 1 // next
		} else {
			{
				pTriInfos[iDstTriIndex + 1].iOrgFaceNumber = f
				pTriInfos[iDstTriIndex + 1].iTSpacesOffs = iTSpacesOffs
			}

			{
				// need an order independent way to evaluate
				// tspace on quads. This is done by splitting
				// along the shortest diagonal.
				i0 := make_index(f, 0)
				i1 := make_index(f, 1)
				i2 := make_index(f, 2)
				i3 := make_index(f, 3)
				T0 := get_tex_coord(pContext, i0)
				T1 := get_tex_coord(pContext, i1)
				T2 := get_tex_coord(pContext, i2)
				T3 := get_tex_coord(pContext, i3)
				distSQ_02 := linalg.length2(T2 - T0)
				distSQ_13 := linalg.length2(T3 - T1)
				bQuadDiagIs_02: bool
				if distSQ_02 < distSQ_13 {
					bQuadDiagIs_02 = true
				} else if distSQ_13 < distSQ_02 {
					bQuadDiagIs_02 = false
				} else {
					P0 := get_position(pContext, i0)
					P1 := get_position(pContext, i1)
					P2 := get_position(pContext, i2)
					P3 := get_position(pContext, i3)
					distSQ_02 = linalg.length2(P2 - P0)
					distSQ_13 = linalg.length2(P3 - P1)

					bQuadDiagIs_02 = distSQ_13 < distSQ_02 ? false : true
				}

				if bQuadDiagIs_02 {
					{
						pVerts_A := &pTriInfos[iDstTriIndex].vert_num
						pVerts_A[0] = 0
						pVerts_A[1] = 1
						pVerts_A[2] = 2
					}
					piTriList_out[iDstTriIndex * 3 + 0] = i0
					piTriList_out[iDstTriIndex * 3 + 1] = i1
					piTriList_out[iDstTriIndex * 3 + 2] = i2
					iDstTriIndex += 1 // next
					{
						pVerts_B := &pTriInfos[iDstTriIndex].vert_num
						pVerts_B[0] = 0
						pVerts_B[1] = 2
						pVerts_B[2] = 3
					}
					piTriList_out[iDstTriIndex * 3 + 0] = i0
					piTriList_out[iDstTriIndex * 3 + 1] = i2
					piTriList_out[iDstTriIndex * 3 + 2] = i3
					iDstTriIndex += 1 // next
				} else {
					{
						pVerts_A := &pTriInfos[iDstTriIndex].vert_num
						pVerts_A[0] = 0
						pVerts_A[1] = 1
						pVerts_A[2] = 3
					}
					piTriList_out[iDstTriIndex * 3 + 0] = i0
					piTriList_out[iDstTriIndex * 3 + 1] = i1
					piTriList_out[iDstTriIndex * 3 + 2] = i3
					iDstTriIndex += 1 // next
					{
						pVerts_B := &pTriInfos[iDstTriIndex].vert_num
						pVerts_B[0] = 1
						pVerts_B[1] = 2
						pVerts_B[2] = 3
					}
					piTriList_out[iDstTriIndex * 3 + 0] = i1
					piTriList_out[iDstTriIndex * 3 + 1] = i2
					piTriList_out[iDstTriIndex * 3 + 2] = i3
					iDstTriIndex += 1 // next
				}
			}
		}

		iTSpacesOffs += verts
		assert(iDstTriIndex <= iNrTrianglesIn)
	}

	for t in 0 ..< iNrTrianglesIn {
		pTriInfos[t].iFlag = {}
	}

	// return total amount of tspaces
	return iTSpacesOffs
}

@(private)
get_position :: proc(pContext: ^Context, index: int) -> Vec3 {
	iF, iI: int
	index_to_data(&iF, &iI, index)
	pos := pContext.interface.get_position(pContext, iF, iI)
	return pos
}

@(private)
get_normal :: proc(pContext: ^Context, index: int) -> Vec3 {
	iF, iI: int
	index_to_data(&iF, &iI, index)
	norm := pContext.interface.get_normal(pContext, iF, iI)
	return norm
}

@(private)
get_tex_coord :: proc(pContext: ^Context, index: int) -> Vec3 {
	iF, iI: int
	res: Vec3
	index_to_data(&iF, &iI, index)
	texc := pContext.interface.get_tex_coord(pContext, iF, iI)
	res.xy = texc.xy
	res.z = 1.0
	return res
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////

@(private)
Edge :: struct #raw_union {
	using a: struct {
		i0, i1, f: int,
	},
	array:   [3]int,
}

// returns the texture area times 2
@(private)
calc_tex_area :: proc(pContext: ^Context, indices: []int) -> f32 {
	t1: Vec3 = get_tex_coord(pContext, indices[0])
	t2: Vec3 = get_tex_coord(pContext, indices[1])
	t3: Vec3 = get_tex_coord(pContext, indices[2])

	t21x := t2.x - t1.x
	t21y := t2.y - t1.y
	t31x := t3.x - t1.x
	t31y := t3.y - t1.y

	fSignedAreaSTx2 := t21x * t31y - t21y * t31x

	return fSignedAreaSTx2 < 0 ? (-fSignedAreaSTx2) : fSignedAreaSTx2
}

@(private)
init_tri_info :: proc(pTriInfos: []Tri_Info, piTriListIn: []int, pContext: ^Context, iNrTrianglesIn: int, allocator := context.allocator) {
	// pTriInfos[f].iFlag is cleared in GenerateInitialVerticesIndexList() which is called before this function.

	// generate neighbor info list
	for f in 0 ..< iNrTrianglesIn {
		for i in 0 ..< 3 {
			pTriInfos[f].FaceNeighbors[i] = -1
			pTriInfos[f].AssignedGroup[i] = nil

			pTriInfos[f].vOs.x = 0.0
			pTriInfos[f].vOs.y = 0.0
			pTriInfos[f].vOs.z = 0.0
			pTriInfos[f].vOt.x = 0.0
			pTriInfos[f].vOt.y = 0.0
			pTriInfos[f].vOt.z = 0.0
			pTriInfos[f].fMagS = 0
			pTriInfos[f].fMagT = 0

			// assumed bad
			pTriInfos[f].iFlag |= {.GroupWithAny}
		}
	}

	// evaluate first order derivatives
	for f in 0 ..< iNrTrianglesIn {
		// initial values
		v1 := get_position(pContext, piTriListIn[f * 3 + 0])
		v2 := get_position(pContext, piTriListIn[f * 3 + 1])
		v3 := get_position(pContext, piTriListIn[f * 3 + 2])
		t1 := get_tex_coord(pContext, piTriListIn[f * 3 + 0])
		t2 := get_tex_coord(pContext, piTriListIn[f * 3 + 1])
		t3 := get_tex_coord(pContext, piTriListIn[f * 3 + 2])

		t21x := t2.x - t1.x
		t21y := t2.y - t1.y
		t31x := t3.x - t1.x
		t31y := t3.y - t1.y
		d1 := v2 - v1
		d2 := v3 - v1

		fSignedAreaSTx2 := t21x * t31y - t21y * t31x

		vOs := (t31y * d1) - (t21y * d2)
		vOt := (-t31x * d1) - (t21x * d2)

		if fSignedAreaSTx2 > 0.0 {
			pTriInfos[f].iFlag |= {.OrientPreserving}
		}

		if fSignedAreaSTx2 != 0 {
			fAbsArea := math.abs(fSignedAreaSTx2)
			fLenOs := linalg.length(vOs)
			fLenOt := linalg.length(vOt)
			fS: f32 = .OrientPreserving in pTriInfos[f].iFlag ? (-1.0) : 1.0
			if fLenOs != 0.0 do pTriInfos[f].vOs = (fS / fLenOs) * vOs
			if fLenOt != 0.0 do pTriInfos[f].vOt = (fS / fLenOt) * vOt

			// evaluate magnitudes prior to normalization of vOs and vOt
			pTriInfos[f].fMagS = fLenOs / fAbsArea
			pTriInfos[f].fMagT = fLenOt / fAbsArea

			// if this is a good triangle
			if pTriInfos[f].fMagS != 0.0 && pTriInfos[f].fMagT != 0.0 {
				pTriInfos[f].iFlag -= {.GroupWithAny}
			}
		}
	}

	t: int

	// force otherwise healthy quads to a fixed orientation
	for (t < (iNrTrianglesIn - 1)) {
		iFO_a := pTriInfos[t].iOrgFaceNumber
		iFO_b := pTriInfos[t + 1].iOrgFaceNumber
		if iFO_a == iFO_b { 	// this is a quad
			bIsDeg_a := .MarkDegenerate in pTriInfos[t].iFlag
			bIsDeg_b := .MarkDegenerate in pTriInfos[t + 1].iFlag

			// bad triangles should already have been removed by
			// DegenPrologue(), but just in case check bIsDeg_a and bIsDeg_a are false
			if bIsDeg_a || bIsDeg_b {
				bOrientA := .OrientPreserving in pTriInfos[t].iFlag
				bOrientB := .OrientPreserving in pTriInfos[t + 1].iFlag
				// if this happens the quad has extremely bad mapping!!
				if bOrientA != bOrientB {
					bChooseOrientFirstTri := false
					if .GroupWithAny in pTriInfos[t + 1].iFlag {
						bChooseOrientFirstTri = true
					} else if calc_tex_area(pContext, piTriListIn[t * 3 + 0:]) >= calc_tex_area(pContext, piTriListIn[(t + 1) * 3 + 0:]) {
						bChooseOrientFirstTri = true
					}

					// force match
					{
						t0 := bChooseOrientFirstTri ? t : (t + 1)
						t1 := bChooseOrientFirstTri ? (t + 1) : t
						pTriInfos[t1].iFlag -= {.OrientPreserving} // clear first
						pTriInfos[t1].iFlag |= (pTriInfos[t0].iFlag & {.OrientPreserving}) // copy bit
					}
				}
			}
			t += 2
		} else {
			t += 1
		}
	}

	// match up edge pairs
	{
		pEdges := make([]Edge, iNrTrianglesIn * 3, allocator)
		defer delete(pEdges)
		if pEdges == nil {
			build_neighbors_slow(pTriInfos, piTriListIn, iNrTrianglesIn)
		} else {
			build_neighbors_fast(pTriInfos, pEdges, piTriListIn, iNrTrianglesIn)
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////

@(private)
build_4_rule_groups :: proc(
	pTriInfos: []Tri_Info,
	pGroups: []Group,
	piGroupTrianglesBuffer: []int,
	piTriListIn: []int,
	iNrTrianglesIn: int,
) -> int {
	iNrMaxGroups := iNrTrianglesIn * 3
	iNrActiveGroups: int
	iOffset: int

	for f in 0 ..< iNrTrianglesIn {
		for i in 0 ..< 3 {
			// if not assigned to a group
			if .GroupWithAny not_in pTriInfos[f].iFlag && pTriInfos[f].AssignedGroup[i] == nil {
				bOrPre: bool
				neigh_indexL, neigh_indexR: int
				vert_index := piTriListIn[f * 3 + i]
				assert(iNrActiveGroups < iNrMaxGroups)
				pTriInfos[f].AssignedGroup[i] = &pGroups[iNrActiveGroups]
				pTriInfos[f].AssignedGroup[i].iVertexRepresentitive = vert_index
				pTriInfos[f].AssignedGroup[i].bOrientPreservering = .OrientPreserving in pTriInfos[f].iFlag
				pTriInfos[f].AssignedGroup[i].iNrFaces = 0
				pTriInfos[f].AssignedGroup[i].pFaceIndices = piGroupTrianglesBuffer[iOffset:]
				iNrActiveGroups += 1

				add_tri_to_group(pTriInfos[f].AssignedGroup[i], f)
				bOrPre = .OrientPreserving in pTriInfos[f].iFlag
				neigh_indexL = pTriInfos[f].FaceNeighbors[i]
				neigh_indexR = pTriInfos[f].FaceNeighbors[i > 0 ? (i - 1) : 2]
				if neigh_indexL >= 0 { 	// neighbor
					bAnswer := assign_recur(piTriListIn, pTriInfos, neigh_indexL, pTriInfos[f].AssignedGroup[i])

					bOrPre2 := .OrientPreserving in pTriInfos[neigh_indexL].iFlag
					bDiff := bOrPre != bOrPre2 ? true : false
					assert(bAnswer || bDiff)
				}
				if neigh_indexR >= 0 { 	// neighbor
					bAnswer := assign_recur(piTriListIn, pTriInfos, neigh_indexR, pTriInfos[f].AssignedGroup[i])

					bOrPre2 := .OrientPreserving in pTriInfos[neigh_indexR].iFlag
					bDiff := bOrPre != bOrPre2 ? true : false
					assert(bAnswer || bDiff)
				}

				// update offset
				iOffset += pTriInfos[f].AssignedGroup[i].iNrFaces
				// since the groups are disjoint a triangle can never
				// belong to more than 3 groups. Subsequently something
				// is completely screwed if this assertion ever hits.
				assert(iOffset <= iNrMaxGroups)
			}
		}
	}

	return iNrActiveGroups
}

@(private)
add_tri_to_group :: proc(pGroup: ^Group, iTriIndex: int) {
	pGroup.pFaceIndices[pGroup.iNrFaces] = iTriIndex
	pGroup.iNrFaces += 1
}

@(private)
assign_recur :: proc(piTriListIn: []int, psTriInfos: []Tri_Info, iMyTriIndex: int, pGroup: ^Group) -> bool {
	pMyTriInfo := &psTriInfos[iMyTriIndex]

	// track down vertex
	iVertRep := pGroup.iVertexRepresentitive
	pVerts := piTriListIn[3 * iMyTriIndex + 0:]
	i := -1
	if pVerts[0] == iVertRep do i = 0
	else if pVerts[1] == iVertRep do i = 1
	else if pVerts[2] == iVertRep do i = 2
	assert(i >= 0 && i < 3)

	// early out
	if pMyTriInfo.AssignedGroup[i] == pGroup do return true
	else if pMyTriInfo.AssignedGroup[i] != nil do return false

	if .GroupWithAny in pMyTriInfo.iFlag {
		// first to group with a group-with-anything triangle
		// determines it's orientation.
		// This is the only existing order dependency in the code!!
		if pMyTriInfo.AssignedGroup[0] == nil && pMyTriInfo.AssignedGroup[1] == nil && pMyTriInfo.AssignedGroup[2] == nil {
			pMyTriInfo.iFlag -= {.OrientPreserving}
			pMyTriInfo.iFlag |= (pGroup.bOrientPreservering ? {.OrientPreserving} : {})
		}
	}

	{
		bOrient := .OrientPreserving in pMyTriInfo.iFlag
		if bOrient != pGroup.bOrientPreservering do return false
	}

	add_tri_to_group(pGroup, iMyTriIndex)
	pMyTriInfo.AssignedGroup[i] = pGroup

	{
		neigh_indexL := pMyTriInfo.FaceNeighbors[i]
		neigh_indexR := pMyTriInfo.FaceNeighbors[i > 0 ? (i - 1) : 2]
		if neigh_indexL >= 0 {
			assign_recur(piTriListIn, psTriInfos, neigh_indexL, pGroup)
		}
		if neigh_indexR >= 0 {
			assign_recur(piTriListIn, psTriInfos, neigh_indexR, pGroup)
		}
	}


	return true
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////

@(private)
generate_t_spaces :: proc(
	psTspace: []T_Space,
	pTriInfos: []Tri_Info,
	pGroups: []Group,
	iNrActiveGroups: int,
	piTriListIn: []int,
	fThresCos: f32,
	pContext: ^Context,
	allocator := context.allocator,
) -> bool {
	iMaxNrFaces, iUniqueTspaces: int
	for g in 0 ..< iNrActiveGroups {
		if iMaxNrFaces < pGroups[g].iNrFaces {
			iMaxNrFaces = pGroups[g].iNrFaces
		}
	}

	if iMaxNrFaces == 0 do return true

	// make initial allocations
	pSubGroupTspace := make([]T_Space, iMaxNrFaces, allocator)
	defer delete(pSubGroupTspace)

	pUniSubGroups := make([]Sub_Group, iMaxNrFaces, allocator)
	defer delete(pUniSubGroups)

	pTmpMembers := make([]int, iMaxNrFaces, allocator)
	defer delete(pTmpMembers)

	if pSubGroupTspace == nil || pUniSubGroups == nil || pTmpMembers == nil {
		return false
	}

	iUniqueTspaces = 0
	for g in 0 ..< iNrActiveGroups {
		pGroup := &pGroups[g]
		iUniqueSubGroups: int

		for i in 0 ..< pGroup.iNrFaces {
			f := pGroup.pFaceIndices[i] // triangle number
			index, iVertIndex, iOF_1 := -1, -1, -1
			iMembers, l: int
			tmp_group: Sub_Group
			bFound: bool
			n, vOs, vOt: Vec3
			if pTriInfos[f].AssignedGroup[0] == pGroup do index = 0
			else if pTriInfos[f].AssignedGroup[1] == pGroup do index = 1
			else if pTriInfos[f].AssignedGroup[2] == pGroup do index = 2
			assert(index >= 0 && index < 3)

			iVertIndex = piTriListIn[f * 3 + index]
			assert(iVertIndex == pGroup.iVertexRepresentitive)

			// is normalized already
			n = get_normal(pContext, iVertIndex)

			// project
			vOs = (pTriInfos[f].vOs - (linalg.dot(n, pTriInfos[f].vOs) * n))
			vOt = (pTriInfos[f].vOt - (linalg.dot(n, pTriInfos[f].vOt) * n))
			if vOs != 0.0 do vOs = linalg.normalize(vOs)
			if vOt != 0.0 do vOt = linalg.normalize(vOt)

			// original face number
			iOF_1 = pTriInfos[f].iOrgFaceNumber

			iMembers = 0
			for j in 0 ..< pGroup.iNrFaces {
				t := pGroup.pFaceIndices[j] // triangle number
				iOF_2 := pTriInfos[t].iOrgFaceNumber

				// project
				vOs2 := pTriInfos[t].vOs - (linalg.dot(n, pTriInfos[t].vOs) * n)
				vOt2 := pTriInfos[t].vOt - (linalg.dot(n, pTriInfos[t].vOt) * n)
				if vOs2 != 0.0 do vOs2 = linalg.normalize(vOs2)
				if vOt2 != 0.0 do vOt2 = linalg.normalize(vOt2)

				{
					bAny := .GroupWithAny in (pTriInfos[f].iFlag | pTriInfos[t].iFlag)
					// make sure triangles which belong to the same quad are joined.
					bSameOrgFace := iOF_1 == iOF_2 ? true : false

					fCosS := linalg.dot(vOs, vOs2)
					fCosT := linalg.dot(vOt, vOt2)

					assert(f != t || bSameOrgFace) // sanity check
					if bAny || bSameOrgFace || (fCosS > fThresCos && fCosT > fThresCos) {
						pTmpMembers[iMembers] = t
						iMembers += 1
					}
				}
			}

			// sort pTmpMembers
			tmp_group.iNrFaces = iMembers
			tmp_group.pTriMembers = pTmpMembers
			if iMembers > 1 {
				uSeed := INTERNAL_RND_SORT_SEED // could replace with a random seed?
				quick_sort(pTmpMembers, 0, iMembers - 1, uSeed)
			}

			// look for an existing match
			bFound = false
			l = 0
			for (l < iUniqueSubGroups && !bFound) {
				bFound = compare_sub_groups(&tmp_group, &pUniSubGroups[l])
				if !bFound do l += 1
			}

			// assign tangent space index
			assert(bFound || l == iUniqueSubGroups)

			// if no match was found we allocate a new subgroup
			if !bFound {
				// insert new subgroup
				pIndices := make([]int, iMembers, allocator)
				if pIndices == nil {
					// clean up and return false
					for s in 0 ..< iUniqueSubGroups {
						delete(pUniSubGroups[s].pTriMembers)
					}
					return false
				}
				pUniSubGroups[iUniqueSubGroups].iNrFaces = iMembers
				pUniSubGroups[iUniqueSubGroups].pTriMembers = pIndices
				copy(pIndices, tmp_group.pTriMembers[:iMembers])
				pSubGroupTspace[iUniqueSubGroups] = eval_t_space(
					tmp_group.pTriMembers,
					iMembers,
					piTriListIn,
					pTriInfos,
					pContext,
					pGroup.iVertexRepresentitive,
				)
				iUniqueSubGroups += 1
			}

			// output tspace
			{
				iOffs := pTriInfos[f].iTSpacesOffs
				iVert := int(pTriInfos[f].vert_num[index])
				pTS_out := &psTspace[iOffs + iVert]
				assert(pTS_out.iCounter < 2)
				assert(.OrientPreserving in pTriInfos[f].iFlag == pGroup.bOrientPreservering)
				if pTS_out.iCounter == 1 {
					pTS_out^ = avg_t_space(pTS_out, &pSubGroupTspace[l])
					pTS_out.iCounter = 2 // update counter
					pTS_out.bOrient = pGroup.bOrientPreservering
				} else {
					assert(pTS_out.iCounter == 0)
					pTS_out^ = pSubGroupTspace[l]
					pTS_out.iCounter = 1 // update counter
					pTS_out.bOrient = pGroup.bOrientPreservering
				}
			}
		}

		// clean up and offset iUniqueTspaces
		for s in 0 ..< iUniqueSubGroups {
			delete(pUniSubGroups[s].pTriMembers)
		}
		iUniqueTspaces += iUniqueSubGroups
	}

	return true
}

@(private)
eval_t_space :: proc(
	face_indices: []int,
	iFaces: int,
	piTriListIn: []int,
	pTriInfos: []Tri_Info,
	pContext: ^Context,
	iVertexRepresentitive: int,
) -> T_Space {
	res: T_Space
	fAngleSum: f32

	for face in 0 ..< iFaces {
		f := face_indices[face]

		// only valid triangles get to add their contribution
		if .GroupWithAny not_in pTriInfos[f].iFlag {
			n, vOs, vOt, p0, p1, p2, v1, v2: Vec3
			fCos, fAngle, fMagS, fMagT: f32
			i, index, i0, i1, i2 := -1, -1, -1, -1, -1
			if piTriListIn[3 * f + 0] == iVertexRepresentitive do i = 0
			else if piTriListIn[3 * f + 1] == iVertexRepresentitive do i = 1
			else if piTriListIn[3 * f + 2] == iVertexRepresentitive do i = 2
			assert(i >= 0 && i < 3)

			// project
			index = piTriListIn[3 * f + i]
			n = get_normal(pContext, index)
			vOs = (pTriInfos[f].vOs - (linalg.dot(n, pTriInfos[f].vOs) * n))
			vOt = (pTriInfos[f].vOt - (linalg.dot(n, pTriInfos[f].vOt) * n))
			if vOs == 0.0 do vOs = linalg.normalize(vOs)
			if vOt == 0.0 do vOt = linalg.normalize(vOt)

			i2 = piTriListIn[3 * f + (i < 2 ? (i + 1) : 0)]
			i1 = piTriListIn[3 * f + i]
			i0 = piTriListIn[3 * f + (i > 0 ? (i - 1) : 2)]

			p0 = get_position(pContext, i0)
			p1 = get_position(pContext, i1)
			p2 = get_position(pContext, i2)
			v1 = p0 - p1
			v2 = p2 - p1

			// project
			v1 = v1 - (linalg.dot(n, v1) * n)
			if v1 != 0.0 do v1 = linalg.normalize(v1)

			v2 = v2 - (linalg.dot(n, v2) * n)
			if v2 != 0.0 do v2 = linalg.normalize(v2)

			// weight contribution by the angle
			// between the two edge vectors
			fCos = linalg.dot(v1, v2)
			fCos = fCos > 1 ? 1 : (fCos < (-1) ? (-1) : fCos)

			fAngle = math.acos(fCos)
			fMagS = pTriInfos[f].fMagS
			fMagT = pTriInfos[f].fMagT

			res.vOs = res.vOs + (fAngle * vOs)
			res.vOt = res.vOt + (fAngle * vOt)
			res.fMagS += (fAngle * fMagS)
			res.fMagT += (fAngle * fMagT)
			fAngleSum += fAngle
		}
	}

	// normalize
	if res.vOs != 0 do res.vOs = linalg.normalize(res.vOs)
	if res.vOt != 0 do res.vOt = linalg.normalize(res.vOt)

	if fAngleSum > 0 {
		res.fMagS /= fAngleSum
		res.fMagT /= fAngleSum
	}

	return res
}

@(private)
compare_sub_groups :: proc(pg1: ^Sub_Group, pg2: ^Sub_Group) -> bool {
	bStillSame := true
	i: int
	if pg1.iNrFaces != pg2.iNrFaces do return false
	for (i < pg1.iNrFaces && bStillSame) {
		bStillSame = pg1.pTriMembers[i] == pg2.pTriMembers[i] ? true : false
		if bStillSame do i += 1
	}
	return bStillSame
}

@(private)
quick_sort :: proc(pSortBuffer: []int, iLeft: int, iRight: int, uSeed: u32) {
	uSeed := uSeed
	iL, iR, n, index, iMid, iTmp: int

	// Random
	t := uSeed & 31
	t = (uSeed << t) | (uSeed >> (32 - t))
	uSeed = uSeed + t + 3
	// Random end

	iL = iLeft
	iR = iRight
	n = (iR - iL) + 1
	assert(n >= 0)
	index = int(uSeed % u32(n))

	iMid = pSortBuffer[index + iL]


	for {
		for (pSortBuffer[iL] < iMid) {
			iL += 1
		}
		for (pSortBuffer[iR] > iMid) {
			iR -= 1
		}

		if iL <= iR {
			iTmp = pSortBuffer[iL]
			pSortBuffer[iL] = pSortBuffer[iR]
			pSortBuffer[iR] = iTmp
			iL += 1
			iR -= 1
		}
		if !(iL <= iR) do break
	}

	if iLeft < iR {
		quick_sort(pSortBuffer, iLeft, iR, uSeed)
	}
	if iL < iRight {
		quick_sort(pSortBuffer, iL, iRight, uSeed)
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

@(private)
build_neighbors_fast :: proc(pTriInfos: []Tri_Info, pEdges: []Edge, piTriListIn: []int, iNrTrianglesIn: int) {
	// build array of edges
	uSeed := INTERNAL_RND_SORT_SEED // could replace with a random seed?
	iEntries0: int
	iCurStartIndex := -1
	for f in 0 ..< iNrTrianglesIn {
		for i in 0 ..< 3 {
			i0 := piTriListIn[f * 3 + i]
			i1 := piTriListIn[f * 3 + (i < 2 ? (i + 1) : 0)]
			pEdges[f * 3 + i].i0 = i0 < i1 ? i0 : i1 // put minimum index in i0
			pEdges[f * 3 + i].i1 = !(i0 < i1) ? i0 : i1 // put maximum index in i1
			pEdges[f * 3 + i].f = f // record face number
		}
	}

	// sort over all edges by i0, this is the pricy one.
	quick_sort_edges(pEdges, 0, iNrTrianglesIn * 3 - 1, 0, uSeed) // sort channel 0 which is i0

	// sub sort over i1, should be fast.
	// could replace this with a 64 bit int sort over (i0,i1)
	// with i0 as msb in the quicksort call above.
	iEntries0 = iNrTrianglesIn * 3
	iCurStartIndex = 0
	for i in 1 ..< iEntries0 {
		if pEdges[iCurStartIndex].i0 != pEdges[i].i0 {
			iL := iCurStartIndex
			iR := i - 1
			//iElems := i-iL
			iCurStartIndex = i
			quick_sort_edges(pEdges, iL, iR, 1, uSeed) // sort channel 1 which is i1
		}
	}

	// sub sort over f, which should be fast.
	// this step is to remain compliant with BuildNeighborsSlow() when
	// more than 2 triangles use the same edge (such as a butterfly topology).
	iCurStartIndex = 0
	for i in 1 ..< iEntries0 {
		if pEdges[iCurStartIndex].i0 != pEdges[i].i0 || pEdges[iCurStartIndex].i1 != pEdges[i].i1 {
			iL := iCurStartIndex
			iR := i - 1
			//iElems := i-iL
			iCurStartIndex = i
			quick_sort_edges(pEdges, iL, iR, 2, uSeed) // sort channel 2 which is f
		}
	}

	// pair up, adjacent triangles
	for i in 0 ..< iEntries0 {
		i0 := pEdges[i].i0
		i1 := pEdges[i].i1
		f := pEdges[i].f
		bUnassigned_A: bool

		i0_A, i1_A: int
		edgenum_A, edgenum_B: int
		get_edge(&i0_A, &i1_A, &edgenum_A, piTriListIn[f * 3:], i0, i1) // resolve index ordering and edge_num
		bUnassigned_A = pTriInfos[f].FaceNeighbors[edgenum_A] == -1 ? true : false

		if bUnassigned_A {
			// get true index ordering
			j := i + 1
			t: int
			bNotFound := true
			for (j < iEntries0 && i0 == pEdges[j].i0 && i1 == pEdges[j].i1 && bNotFound) {
				bUnassigned_B: bool
				i0_B, i1_B: int
				t = pEdges[j].f
				// flip i0_B and i1_B
				get_edge(&i1_B, &i0_B, &edgenum_B, piTriListIn[t * 3:], pEdges[j].i0, pEdges[j].i1) // resolve index ordering and edge_num

				bUnassigned_B = pTriInfos[t].FaceNeighbors[edgenum_B] == -1 ? true : false
				if i0_A == i0_B && i1_A == i1_B && bUnassigned_B {
					bNotFound = false
				} else {
					j += 1
				}
			}

			if !bNotFound {
				t = pEdges[j].f
				pTriInfos[f].FaceNeighbors[edgenum_A] = t
				pTriInfos[t].FaceNeighbors[edgenum_B] = f
			}
		}
	}
}

@(private)
build_neighbors_slow :: proc(pTriInfos: []Tri_Info, piTriListIn: []int, iNrTrianglesIn: int) {
	for f in 0 ..< iNrTrianglesIn {
		for i in 0 ..< 3 {
			// if unassigned
			if pTriInfos[f].FaceNeighbors[i] == -1 {
				i0_A := piTriListIn[f * 3 + i]
				i1_A := piTriListIn[f * 3 + (i < 2 ? (i + 1) : 0)]

				// search for a neighbor
				bFound := false
				t, j: int
				for (!bFound && t < iNrTrianglesIn) {
					if t != f {
						j = 0
						for (!bFound && j < 3) {
							// in rev order
							i1_B := piTriListIn[t * 3 + j]
							i0_B := piTriListIn[t * 3 + (j < 2 ? (j + 1) : 0)]
							if i0_A == i0_B && i1_A == i1_B {
								bFound = true
							} else {
								j += 1
							}
						}
					}

					if !bFound do t += 1
				}

				// assign neighbors
				if bFound {
					pTriInfos[f].FaceNeighbors[i] = t
					pTriInfos[t].FaceNeighbors[j] = f
				}
			}
		}
	}
}

@(private)
quick_sort_edges :: proc(pSortBuffer: []Edge, iLeft: int, iRight: int, channel: int, uSeed: u32) {
	uSeed := uSeed
	t: u32
	iL, iR, n, index, iMid: int

	// early out
	sTmp: Edge
	iElems := iRight - iLeft + 1
	if iElems < 2 {
		return
	} else if iElems == 2 {
		if pSortBuffer[iLeft].array[channel] > pSortBuffer[iRight].array[channel] {
			sTmp = pSortBuffer[iLeft]
			pSortBuffer[iLeft] = pSortBuffer[iRight]
			pSortBuffer[iRight] = sTmp
		}
		return
	}

	// Random
	t = uSeed & 31
	t = (uSeed << t) | (uSeed >> (32 - t))
	uSeed = uSeed + t + 3
	// Random end

	iL = iLeft
	iR = iRight
	n = (iR - iL) + 1
	assert(n >= 0)
	index = int(uSeed % u32(n))

	iMid = pSortBuffer[index + iL].array[channel]

	for {
		for (pSortBuffer[iL].array[channel] < iMid) {
			iL += 1
		}

		for (pSortBuffer[iR].array[channel] > iMid) {
			iR -= 1
		}

		if iL <= iR {
			sTmp = pSortBuffer[iL]
			pSortBuffer[iL] = pSortBuffer[iR]
			pSortBuffer[iR] = sTmp
			iL += 1
			iR -= 1
		}
		if !(iL <= iR) do break
	}

	if iLeft < iR {
		quick_sort_edges(pSortBuffer, iLeft, iR, channel, uSeed)
	}
	if iL < iRight {
		quick_sort_edges(pSortBuffer, iL, iRight, channel, uSeed)
	}
}

// resolve ordering and edge number
@(private)
get_edge :: proc(i0_out: ^int, i1_out: ^int, edgenum_out: ^int, indices: []int, i0_in: int, i1_in: int) {
	edgenum_out^ = -1

	// test if first index is on the edge
	if indices[0] == i0_in || indices[0] == i1_in {
		// test if second index is on the edge
		if indices[1] == i0_in || indices[1] == i1_in {
			edgenum_out^ = 0 // first edge
			i0_out^ = indices[0]
			i1_out^ = indices[1]
		} else {
			edgenum_out^ = 2 // third edge
			i0_out^ = indices[2]
			i1_out^ = indices[0]
		}
	} else {
		// only second and third index is on the edge
		edgenum_out^ = 1 // second edge
		i0_out^ = indices[1]
		i1_out^ = indices[2]
	}
}


/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// Degenerate triangles ////////////////////////////////////

@(private)
degen_prologue :: proc(pTriInfos: []Tri_Info, piTriList_out: []int, iNrTrianglesIn: int, iTotTris: int) {
	iNextGoodTriangleSearchIndex := -1
	bStillFindingGoodOnes: bool

	// locate quads with only one good triangle
	t: int
	for t < (iTotTris - 1) {
		iFO_a := pTriInfos[t].iOrgFaceNumber
		iFO_b := pTriInfos[t + 1].iOrgFaceNumber
		if iFO_a == iFO_b { 	// this is a quad
			bIsDeg_a := .MarkDegenerate in pTriInfos[t].iFlag
			bIsDeg_b := .MarkDegenerate in pTriInfos[t + 1].iFlag

			if bIsDeg_a != bIsDeg_b {
				pTriInfos[t].iFlag |= {.QuadOneDegenTri}
				pTriInfos[t + 1].iFlag |= {.QuadOneDegenTri}
			}
			t += 2
		} else {
			t += 1
		}
	}

	// reorder list so all degen triangles are moved to the back
	// without reordering the good triangles
	iNextGoodTriangleSearchIndex = 1
	t = 0
	bStillFindingGoodOnes = true
	for (t < iNrTrianglesIn && bStillFindingGoodOnes) {
		bIsGood := .MarkDegenerate not_in pTriInfos[t].iFlag
		if bIsGood {
			if iNextGoodTriangleSearchIndex < (t + 2) {
				iNextGoodTriangleSearchIndex = t + 2
			}
		} else {
			t0, t1: int
			// search for the first good triangle.
			bJustADegenerate := true
			for (bJustADegenerate && iNextGoodTriangleSearchIndex < iTotTris) {
				bIsGood = .MarkDegenerate not_in pTriInfos[iNextGoodTriangleSearchIndex].iFlag
				if bIsGood {
					bJustADegenerate = false
				} else {
					iNextGoodTriangleSearchIndex += 1
				}

			}

			t0 = t
			t1 = iNextGoodTriangleSearchIndex
			iNextGoodTriangleSearchIndex += 1
			assert(iNextGoodTriangleSearchIndex > (t + 1))

			// swap triangle t0 and t1
			if !bJustADegenerate {
				for i in 0 ..< 3 {
					index := piTriList_out[t0 * 3 + i]
					piTriList_out[t0 * 3 + i] = piTriList_out[t1 * 3 + i]
					piTriList_out[t1 * 3 + i] = index
				}
				{
					tri_info: Tri_Info = pTriInfos[t0]
					pTriInfos[t0] = pTriInfos[t1]
					pTriInfos[t1] = tri_info
				}
			} else {
				bStillFindingGoodOnes = false // this is not supposed to happen
			}
		}

		if bStillFindingGoodOnes do t += 1
	}

	assert(bStillFindingGoodOnes) // code will still work.
	assert(iNrTrianglesIn == t)
}

@(private)
degen_epilogue :: proc(
	psTspace: []T_Space,
	pTriInfos: []Tri_Info,
	piTriListIn: []int,
	pContext: ^Context,
	iNrTrianglesIn: int,
	iTotTris: int,
) {
	// deal with degenerate triangles
	// punishment for degenerate triangles is O(N^2)
	for t in iNrTrianglesIn ..< iTotTris {
		// degenerate triangles on a quad with one good triangle are skipped
		// here but processed in the next loop
		bSkip := .QuadOneDegenTri in pTriInfos[t].iFlag

		if !bSkip {
			for i in 0 ..< 3 {
				index1 := piTriListIn[t * 3 + i]
				// search through the good triangles
				bNotFound := true
				j: int
				for (bNotFound && j < (3 * iNrTrianglesIn)) {
					index2 := piTriListIn[j]
					if index1 == index2 {
						bNotFound = false
					} else {
						j += 1
					}
				}

				if !bNotFound {
					iTri := j / 3
					iVert := j % 3
					iSrcVert := int(pTriInfos[iTri].vert_num[iVert])
					iSrcOffs := pTriInfos[iTri].iTSpacesOffs
					iDstVert := int(pTriInfos[t].vert_num[i])
					iDstOffs := pTriInfos[t].iTSpacesOffs

					// copy tspace
					psTspace[iDstOffs + iDstVert] = psTspace[iSrcOffs + iSrcVert]
				}
			}
		}
	}

	// deal with degenerate quads with one good triangle
	for t in 0 ..< iNrTrianglesIn {
		// this triangle belongs to a quad where the
		// other triangle is degenerate
		if .QuadOneDegenTri in pTriInfos[t].iFlag {
			vDstP: Vec3
			iOrgF := -1
			bNotFound: bool
			pV := &pTriInfos[t].vert_num
			iFlag := (1 << pV[0]) | (1 << pV[1]) | (1 << pV[2])
			iMissingIndex := 0
			if (iFlag & 2) == 0 do iMissingIndex = 1
			else if (iFlag & 4) == 0 do iMissingIndex = 2
			else if (iFlag & 8) == 0 do iMissingIndex = 3

			iOrgF = pTriInfos[t].iOrgFaceNumber
			vDstP = get_position(pContext, make_index(iOrgF, iMissingIndex))
			bNotFound = true
			i := 0

			for (bNotFound && i < 3) {
				iVert := int(pV[i])
				vSrcP: Vec3 = get_position(pContext, make_index(iOrgF, iVert))
				if vSrcP == vDstP {
					iOffs := pTriInfos[t].iTSpacesOffs
					psTspace[iOffs + iMissingIndex] = psTspace[iOffs + iVert]
					bNotFound = false
				} else {
					i += 1
				}
			}
			assert(!bNotFound)
		}
	}
}
