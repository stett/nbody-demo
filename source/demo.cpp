#include "demo.h"
#include <cmath>
#include <random>
#include "cinder/CinderImGui.h"
#include "cinder/Display.h"

using namespace ci;
using namespace ci::gl;

void nbody::Demo::setup()
{
    ImGui::Initialize();

    {
        // For high-dpi displays
        const float content_scale = getDisplay()->getContentScale();
        ImGuiIO &io = ImGui::GetIO();
        io.DisplayFramebufferScale.x *= content_scale;
        io.DisplayFramebufferScale.y *= content_scale;
    }

    setWindowSize(1024, 1024);

    //rng.seed(42);

    time = getElapsedSeconds();

    {
        static const std::string shader_vert(
                "#version 410\n"
                "layout (location=0) in vec3 v_min_pos;"
                "layout (location=1) in vec3 v_max_pos;"
                "layout (location=2) in float v_potential;"
                "layout (location=0) out vec3 g_min_pos;"
                "layout (location=1) out vec3 g_max_pos;"
                "layout (location=2) out float g_potential;"
                "void main(void)"
                "{"
                "    g_min_pos = v_min_pos;"
                "    g_max_pos = v_max_pos;"
                "    g_potential = v_potential;"
                "}");
        static const std::string shader_geom(
                "#version 410\n"
                "uniform mat4	ciModelViewProjection;"
                "layout (points) in;"
                "layout (line_strip, max_vertices = 18) out;"
                "layout (location=0) in vec3 g_min_pos[];"
                "layout (location=1) in vec3 g_max_pos[];"
                "layout (location=2) in float g_potential[];"
                "layout (location=0) out float f_potential;"
                "void main(void)"
                "{"
                "    f_potential = g_potential[0];"
                "    vec3 aa = g_min_pos[0];"
                "    vec3 bb = g_max_pos[0];"
                "    vec4 v[8] = vec4[8]("
                "       ciModelViewProjection * vec4(aa.x, aa.y, aa.z, 1),"
                "       ciModelViewProjection * vec4(bb.x, aa.y, aa.z, 1),"
                "       ciModelViewProjection * vec4(bb.x, bb.y, aa.z, 1),"
                "       ciModelViewProjection * vec4(aa.x, bb.y, aa.z, 1),"
                "       ciModelViewProjection * vec4(aa.x, aa.y, bb.z, 1),"
                "       ciModelViewProjection * vec4(bb.x, aa.y, bb.z, 1),"
                "       ciModelViewProjection * vec4(bb.x, bb.y, bb.z, 1),"
                "       ciModelViewProjection * vec4(aa.x, bb.y, bb.z, 1));"
                ""
                "    gl_Position = v[0]; EmitVertex();"
                "    gl_Position = v[0+4]; EmitVertex();"
                "    gl_Position = v[0]; EmitVertex();"
                "    gl_Position = v[1]; EmitVertex();"
                "    gl_Position = v[1+4]; EmitVertex();"
                "    gl_Position = v[1]; EmitVertex();"
                "    gl_Position = v[2]; EmitVertex();"
                "    gl_Position = v[2+4]; EmitVertex();"
                "    gl_Position = v[2]; EmitVertex();"
                "    gl_Position = v[3]; EmitVertex();"
                "    gl_Position = v[3+4]; EmitVertex();"
                "    gl_Position = v[3]; EmitVertex();"
                "    gl_Position = v[0]; EmitVertex();"
                "    gl_Position = v[4]; EmitVertex();"
                "    gl_Position = v[5]; EmitVertex();"
                "    gl_Position = v[6]; EmitVertex();"
                "    gl_Position = v[7]; EmitVertex();"
                "    gl_Position = v[4]; EmitVertex();"
                "    EndPrimitive();"
                "}"
        );
        static const std::string shader_frag(
                "#version 410\n"
                "out vec4 		oColor;"
                "layout (location=0) in float f_potential;"
                "void main(void)"
                "{"
                "    float percent = f_potential;"
                "    float a = percent;"
                "    vec4 potential_color = vec4(1-(a*.5), a, 0, .1 + .15 * a);"
                "    oColor = potential_color;"
                "}");
        bounds_shader = gl::GlslProg::create(shader_vert, shader_frag, shader_geom);
    }

    {
        static const std::string shader_vert(
                "#version 410\n"
                "uniform mat4 ciModelViewProjection;"
                "layout (location=0) in vec3 v_pos;"
                "layout (location=1) in float v_rad;"
                //"layout (location=2) in vec4 v_rot;"
                "layout (location=0) out float g_rad;"
                "void main(void)"
                "{"
                "    gl_Position = ciModelViewProjection * vec4(v_pos, 1);"
                "    g_rad = v_rad;"
                "}");
        static const std::string shader_geom(
                "#version 410\n"
                "#define pi 3.1415926535897932384626433832795\n"
                "uniform mat4 ciProjectionMatrix;"
                //"uniform float radius;"
                "layout (points) in;"
                "layout (triangle_strip, max_vertices = 38) out;"
                "layout (location=0) in float g_rad[];"
                "void main(void)"
                "{"
                "    float aspect = ciProjectionMatrix[1][1] / ciProjectionMatrix[0][0];"
                //"    vec2 scale = vec2(1, aspect);"
                "    vec2 scale = vec2(1, aspect) * ciProjectionMatrix[0][0];"
                "    float r = g_rad[0];"
                //"    float r = radius;"
                "    float n = 20;"
                "    gl_Position = gl_in[0].gl_Position + vec4(r*scale.x,0, 0, 0);"
                "    EmitVertex();"
                "    for (float i = 1; i < n; ++i) {"
                "        float t = (i / (n-1)) * 2 * pi;"
                "        vec2 xy = r * vec2(cos(t), sin(t));"
                "        gl_Position = gl_in[0].gl_Position + vec4(xy.x*scale.x, xy.y*scale.y, 0, 0);"
                "        EmitVertex();"
                "        gl_Position = gl_in[0].gl_Position;"
                "        EmitVertex();"
                "    }"
                "    EndPrimitive();"
                "}");
        static const std::string shader_frag(
                "#version 150\n"
                "out vec4 oColor;"
                "void main(void)"
                "{"
                "    oColor = vec4(1,1,1,1);"
                "}");
        particle_shader = gl::GlslProg::create(shader_vert, shader_frag, shader_geom);
    }

    setup_sim_data();

    // Create and populate VBOs containing particle and bounds data
    vbo_particles = gl::Vbo::create(GL_ARRAY_BUFFER, sim.bodies.size() * 3, nullptr, GL_DYNAMIC_DRAW);
    vbo_bounds = gl::Vbo::create(GL_ARRAY_BUFFER, sim.acc_tree.nodes().size() * 7, nullptr, GL_DYNAMIC_DRAW);
    update_gpu_data();

    gl::enableDepthWrite();
    gl::enableDepthRead();
}

void nbody::Demo::spawn_galaxy(uint32_t num, nbody::util::DiskArgs args)
{
    sim.bodies.resize(sim.bodies.size() + num);
    nbody::util::disk(sim.bodies.end() - num, sim.bodies.end(), args);
}

void nbody::Demo::spawn_cube(uint32_t num, nbody::util::CubeArgs args)
{
    sim.bodies.resize(sim.bodies.size() + num);
    nbody::util::cube(sim.bodies.end() - num, sim.bodies.end(), args);
}

void nbody::Demo::setup_sim_data()
{
    // remove all bodies from the sim
    sim.bodies.clear();

    // fill the void with evenly spaced stars
    //spawn_cube(target_num_elems, { .size=sim.size });
    //spawn_cube(target_num_elems, { .size=1000 });


    // add a disk galaxy at the origin
    spawn_galaxy(target_num_elems, { .center={0,0,0}, .axis={0,0,1}, .vel={0,0,0} });

    //spawn_galaxy(target_num_elems, { .center={-250,0,0}, .axis={0,0,1}, .vel={0,40,0} });
    //spawn_galaxy(target_num_elems, { .center={250,0,0},  .axis={0,1,0}, .vel={0,-40,0} });

    //spawn_galaxy(target_num_elems, { .center={-500,0,0}, .axis={0,0,1}, .vel={0,0,0} });
    //spawn_galaxy(target_num_elems, { .center={300,0,0}, .axis={0,1,0}, .vel={0,0,.001} });

    // this forces an update to the acceleration structure, which is
    // needed if we want to update the structure rendering
    sim.accelerate();
}

void nbody::Demo::update_gpu_data()
{
    // Update the CPU buffer for particle data
    gpu_particle_data.resize(sim.bodies.size() * 8);
    for (size_t i = 0; i < sim.bodies.size(); i++)
    {
        nbody::Body& body = sim.bodies[i];
        gpu_particle_data[(i * 4) + 0] = (body.pos.x);
        gpu_particle_data[(i * 4) + 1] = (body.pos.y);
        gpu_particle_data[(i * 4) + 2] = (body.pos.z);
        gpu_particle_data[(i * 4) + 3] = (body.radius);
    }

    // Update the GPU buffer
    vbo_particles->bufferData(gpu_particle_data.size() * sizeof(float), gpu_particle_data.data(), GL_DYNAMIC_DRAW);

    if (draw_bh_bounds)
    {
        // Update the CPU buffer for tree data
        // Create and populate VBO containing bounds data
        const size_t num_nodes = sim.acc_tree.nodes().size();
        gpu_bounds_data.clear();
        gpu_bounds_data.reserve(7 * num_nodes);
        float max_potential = 0;
        float avg_potential = 0;
        const float num_nodes_inv = 1.f / float(num_nodes);
        for (const nbody::bh::Node& node : sim.acc_tree.nodes())
        {
            const vec3 half = vec3(node.bounds.size * .5f);
            const vec3 bounds_center = vec3(node.bounds.center.x, node.bounds.center.y, node.bounds.center.z);
            for (size_t i = 0; i < 3; ++i)
                gpu_bounds_data.emplace_back(bounds_center[i] - half[i]);
            for (size_t i = 0; i < 3; ++i)
                gpu_bounds_data.emplace_back(bounds_center[i] + half[i]);

            // get gravitational potential at the center of this node and store it in GPU data
            const nbody::Vector& center = node.bounds.center;
            float potential = 0;
            sim.acc_tree.apply(center, [&potential, &center](const nbody::bh::Node& node) {
                const vec3 delta = vec3(node.com.x, node.com.y, node.com.z) - vec3(center.x, center.y, center.z);
                potential += node.mass / dot(delta, delta);
            });
            max_potential = std::max(max_potential, potential);
            avg_potential += potential * num_nodes_inv;
            gpu_bounds_data.emplace_back(potential);
        }
        const float max_potential_inv = max_potential > std::numeric_limits<float>::epsilon() ? 1.f / max_potential : 0;
        const float avg_potential_inv = avg_potential > std::numeric_limits<float>::epsilon() ? 1.f / avg_potential : 0;
        for (size_t i = 0; i < gpu_bounds_data.size(); i += 7)
        {
            float& potential = gpu_bounds_data[i+6];
            potential = std::min(1.f, potential * avg_potential_inv);
        }
        vbo_bounds->bufferData(gpu_bounds_data.size() * sizeof(float), gpu_bounds_data.data(), GL_DYNAMIC_DRAW);
    }
}

void nbody::Demo::update_selected_body()
{
    /*
    vec3 glm_ray_origin;
    vec3 glm_ray_direction;
    mouse_ray(glm_ray_origin, glm_ray_direction);
    const bh3::Vector ray_origin = { glm_ray_origin.x, glm_ray_origin.y, glm_ray_origin.z };
    const bh3::Vector ray_direction = { glm_ray_direction.x, glm_ray_direction.y, glm_ray_direction.z };
    sim.bhtree.ray_query(ray_origin, ray_direction, [this](const bh3::Node& node)
    {
        // if this node has children, keep digging
        if (node.children > 0)
            return true;

        // if this node is a child, test against the element in the node
        sim.bodies[node.]
    });
     */
}

void nbody::Demo::resize()
{
    camera.setPerspective(60, getWindowAspectRatio(), 1, 1e5 );
    gl::setMatrices(camera );
}

void nbody::Demo::update()
{
    bool one_tick = false;

    // Update gui
    {
        ImGui::Begin("Settings");
        int app_hz = int(floor(1.f / delta_time));
        ImGui::Text("framerate: %dhz", app_hz);
        const int bhtree_percent = int(100.f * float(sim.acc_tree.nodes().size()) / float(sim.acc_tree.nodes().capacity()));
        ImGui::Text("node capacity: %d (%d%%)", (int)sim.acc_tree.nodes().size(), bhtree_percent);
        ImGui::Checkbox("run simulation", &run_simulation);
        int sim_hz = int(ceil(1.f / sim_dt));
        if (ImGui::SliderInt("sim hz", &sim_hz, 1.f, 120.f)) { sim_dt = 1.f / float(sim_hz); }
        if (ImGui::SliderFloat("sim t-scale", &sim_dt_scale, .0f, 1.f)) { }
        if (ImGui::Button("tick simulation")) { one_tick = true; }
        if (ImGui::Button("reset simulation")) { setup_sim_data(); }
#if NBODY_GPU
        if (ImGui::Checkbox("gpu acceleration", &sim.use_gpu)) { }
#endif
        if (ImGui::Checkbox("show gravity tree", &draw_bh_bounds)) { }
        if (ImGui::Checkbox("show coordinate axes", &draw_axes)) { }

        ImGui::Text("# stars: %d", int(target_num_elems));
        int log_num_stars = std::log2(target_num_elems);
        if (ImGui::SliderInt("log2(# stars)", &log_num_stars, 0, 18))
        {
            target_num_elems = std::pow(2, log_num_stars);
            setup_sim_data();
        }
        ImGui::End();
    }

    // Update dt
    double new_time = getElapsedSeconds();
    delta_time = float(new_time - time);
    time = new_time;

    // Update camera
    {
        cam_focus_target = vec3(0);

        const float snap = 2.f;
        cam_focus += (cam_focus_target - cam_focus) * std::min(delta_time * snap, 1.f);
        cam_angles += (cam_target_angles - cam_angles) * std::min(delta_time * snap, 1.f);
        cam_dist += (cam_target_dist - cam_dist) * std::min(delta_time * snap, 1.f);
        const glm::mat4 m0 = glm::rotate(cam_angles[0], glm::vec3(0, 1, 0));
        const glm::mat4 m1 = glm::rotate(cam_angles[1], glm::vec3(0, 0, 1));
        const glm::mat4 m2 = glm::translate(vec3(cam_dist, 0, 0));
        // NOTE: The cam_focus bit is really not right
        const glm::vec3 cam_pos = m0 * m1 * m2 * glm::translate(cam_focus) * vec4(0,0,0, 1);
        camera.lookAt(cam_pos, cam_focus, vec3(0, 1, 0));
    }

    // if the last delta tick was too big, stop running the sim
    if (run_simulation && delta_time > .5)
        run_simulation = false;

    if (one_tick)
    {
        one_tick = false;
        sim.update(sim_dt);
    }
    else {
        // if running simulation, tick it
        if (run_simulation) {
            sim_dt_accum += delta_time;
            size_t sim_steps = 1;
            while (sim_dt_accum > sim_dt && sim_steps-- > 0) {
                sim_dt_accum -= sim_dt;
                sim.update(sim_dt * sim_dt_scale);
            }
        }
    }

    // Update GPU data
    update_gpu_data();
}

void nbody::Demo::mouseMove(MouseEvent event)
{
    mouse_pos = event.getPos();
}

void nbody::Demo::mouseDrag(MouseEvent event)
{
    const glm::ivec2 new_mouse_pos = event.getPos();
    mouse_delta = new_mouse_pos - mouse_pos;
    mouse_pos = new_mouse_pos;

    if (!mouse_drag)
    {
        cam_target_angles[0] -= mouse_delta.x * .01f;
        cam_target_angles[1] += mouse_delta.y * .01f;

        if (cam_target_angles[1] < -M_PI * .4f) { cam_target_angles[1] = -M_PI * .4f; }
        if (cam_target_angles[1] > M_PI * .4f) { cam_target_angles[1] = M_PI * .4f; }
    }
}

void nbody::Demo::mouseWheel(MouseEvent event)
{
    cam_target_dist -= event.getWheelIncrement() * 5.f;
    if (cam_target_dist < 1.f) { cam_target_dist = 1.f; }
}

void nbody::Demo::mouseDown(MouseEvent event)
{
    // shift click spawns a new galaxy
    if (event.isShiftDown())
    {
        mouse_world_drag_origin = mouse_world_pos();
        mouse_drag = true;
        /*
        const vec3 mp = mouse_world_pos();
        const nbody::Vector pos = {mp.x, mp.y, mp.z};
        const nbody::Vector axis = {0,0,1};
        spawn_galaxy(pos, axis, target_num_elems);
         */
    }
}

void nbody::Demo::mouseUp(MouseEvent event)
{
    if (mouse_drag)
    {
        mouse_drag = false;
        const vec3 pos0 = mouse_world_drag_origin;
        const vec3 pos1 = mouse_world_pos();
        const vec3 diff = pos1 - mouse_world_drag_origin;
        const vec3 n = normalize(diff);
        const nbody::Vector galaxy_axis = {n.x, n.y, n.z};
        const nbody::Vector galaxy_vel = galaxy_axis * length(diff) * .00001f;
        const nbody::Vector galaxy_pos = {pos0.x, pos0.y, pos0.z};
        spawn_galaxy(target_num_elems, {.center=galaxy_pos, .vel=galaxy_vel, .axis=galaxy_axis });
    }
}

void nbody::Demo::draw()
{
    gl::clear(ColorA(0, 0, 0, 1), true);

    gl::setMatrices(camera);

    if (mouse_drag)
    {
        const vec3 pos0 = mouse_world_drag_origin;
        const vec3 pos1 = mouse_world_pos();
        gl::color(1, .2, .2, .9);
        gl::drawLine(vec3(0), vec3(pos1.x, 0, 0));
        gl::color(.2, 1, .2, .9);
        gl::drawLine(vec3(pos1.x, 0, 0), vec3(pos1.x, pos1.y, 0));
        gl::color(.4, .4, 1, .9);
        gl::drawLine(vec3(pos1.x, pos1.y, 0), pos1);

        gl::color(1,1,0,1);
        gl::drawLine(pos0, pos1);
    }


    if (draw_bh_bounds)
    {
        gl::ScopedGlslProg glsl_scope(bounds_shader);
        gl::ScopedDepth depth_scope(false);
        vbo_bounds->bind();
        gl::enableVertexAttribArray(0);
        gl::enableVertexAttribArray(1);
        gl::enableVertexAttribArray(2);
        gl::vertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7*sizeof(float), (void*)(0*sizeof(float)));
        gl::vertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7*sizeof(float), (void*)(3*sizeof(float)));
        gl::vertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 7*sizeof(float), (void*)(6*sizeof(float)));
        gl::drawArrays(GL_POINTS, 0, (GLsizei)sim.acc_tree.nodes().size());
        vbo_bounds->unbind();
        gl::setDefaultShaderVars();
    }

    if (draw_axes)
    {
        gl::color(1, .2, .2, .5);
        gl::drawLine(vec3(-sim.size, 0, 0), vec3(sim.size, 0, 0));
        gl::color(.2, 1, .2, .5);
        gl::drawLine(vec3(0, -sim.size, 0), vec3(0, sim.size, 0));
        gl::color(.2, .2, 1, .5);
        gl::drawLine(vec3(0, 0, -sim.size), vec3(0, 0, sim.size));
    }

    {
        gl::ScopedGlslProg glsl_scope(particle_shader);
        vbo_particles->bind();
        gl::enableVertexAttribArray(0);
        gl::enableVertexAttribArray(1);
        gl::vertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4*sizeof(float), nullptr);
        gl::vertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(3*sizeof(float)));
        gl::drawArrays(GL_POINTS, 0, (GLsizei) sim.bodies.size());
        vbo_particles->unbind();
        gl::setDefaultShaderVars();
    }

    /*
    if (draw_selection)
    {
        if (0 <= selected_elem && selected_elem < sim.bodies.size())
        {
            gl::color(1, 0, 0, 1);
            const nbody::Vector& nbpos = sim.bodies[selected_elem].pos;
            const vec3 pos = vec3(nbpos.x, nbpos.y, nbpos.z);
            gl::drawSphere(pos, particle_radius * 2);

            size_t num_interactions = 0;
            sim.acc_tree.apply({ pos.x, pos.y, pos.z }, [&](const nbody::bh::Node& node)
			{
				++num_interactions;

                // select a color based on tree depth
				const float percent = 1.f - (node.bounds.size / sim.bhtree.bounds().size);
				const float a = percent * percent;
				gl::color(1 - a * .5, a, 0, .25 + .75 * a);

                // draw line from selected element to this node's com
                const vec3& com = vec3(node.com.x, node.com.y, node.com.z);
				gl::drawLine(pos, com);

                // if this node doesn't have children, outline it and draw crosshairs
                // at the center of mass.
                if (node.children != 0)
                {
                    const float bounds_size = node.bounds.size;
                    const bh3::Vector bounds_center = node.bounds.center;
                    gl::drawStrokedCube(vec3(bounds_center.x, bounds_center.y, bounds_center.z), vec3(bounds_size));
                    gl::drawLine(com - vec3(bounds_size * .25, 0, 0), com + vec3(bounds_size * .25, 0, 0));
                    gl::drawLine(com - vec3(0, bounds_size * .25, 0), com + vec3(0, bounds_size * .25, 0));
                    gl::drawLine(com - vec3(0, 0, bounds_size * .25), com + vec3(0, 0, bounds_size * .25));
                }
			});
        }
    }
     */

    /*
    if (draw_collisions)
    {
        gl::color(1,0,0,1);
        for (uint32_t i = 0; i < sim.sbvhtree.num_intersections; ++i)
        {
            const uint32_t i0 = sim.sbvhtree.intersections[i].first;
            const uint32_t i1 = sim.sbvhtree.intersections[i].second;
            const nbody::Body& body0 = sim.bodies[i0];
            const nbody::Body& body1 = sim.bodies[i1];
            gl::drawLine(body0.pos, body1.pos);
        }
    }
     */
}

vec3 nbody::Demo::homogeneous_to_world(const vec3& homo) const
{
    const mat4 view = camera.getViewMatrix();
    const mat4 proj = camera.getProjectionMatrix();
    const vec4 world = glm::inverse(proj * view) * vec4(homo, 1.f);
    return vec3(world) / world.w;
}

void nbody::Demo::mouse_ray(vec3& out_ray_origin, vec3& out_ray_direction) const
{
    const vec2 mouse_homo = vec2(
            2.0f * (float)mouse_pos.x / (float)getWindowWidth() - 1.0f,
            1.0f - 2.0f * (float)mouse_pos.y / (float)getWindowHeight());
    out_ray_origin = camera.getEyePoint();
    out_ray_direction = normalize(homogeneous_to_world(vec3(mouse_homo, 0)) - out_ray_origin);

    //out_ray_origin = homogeneous_to_world(vec3(mouse_homo, 0));
    //const vec3 ray_end = homogeneous_to_world(vec3(mouse_homo, 1));
    //out_ray_direction = glm::normalize(ray_end - out_ray_origin);
}

vec3 nbody::Demo::mouse_plane_pos(const vec3& plane_point, const vec3& plane_axis) const
{
    vec3 ray_origin;
    vec3 ray_direction;
    mouse_ray(ray_origin, ray_direction);

    const vec3 diff = ray_origin - plane_point;
    float numer = dot(diff, plane_axis);
    float denom = dot(ray_direction, plane_axis);
    if (std::numeric_limits<float>::epsilon() > denom && denom > -std::numeric_limits<float>::epsilon()) {
        return plane_point;
    }
    const float t = numer / denom;
    const vec3 proj = ray_origin - (ray_direction * t);
    return proj;
}

vec3 nbody::Demo::mouse_world_pos(const float dist_from_eye) const
{
    vec3 ray_origin;
    vec3 ray_direction;
    mouse_ray(ray_origin, ray_direction);
    const vec3 plane_pos = camera.getEyePoint() + (ray_direction * dist_from_eye);
    const vec3 plane_axis = -ray_direction;
    return mouse_plane_pos(plane_pos, plane_axis);
}
