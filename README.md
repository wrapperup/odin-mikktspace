# Odin MikkTSpace

Port of [Morten S. Mikkelsen's Tangent Space algorithm](https://github.com/mmikk/MikkTSpace) in Odin.

## Usage

```odin
import mikk "mikktspace"

mesh_data := ...

interface := mikk.Interface {
    get_num_faces            = get_num_faces,
    get_num_vertices_of_face = get_num_vertices_of_face,
    get_position             = get_position,
    get_normal               = get_normal,
    get_tex_coord            = get_tex_coord,
    set_t_space_basic        = set_t_space_basic,
}

ctx := mikk.Context {
    interface = &interface,
    user_data  = &mesh_data,
}

ok := mikk.generate_tangents(&ctx)
```

[Full example can be found here](example/example.odin)

## License
[zlib License](LICENSE)
