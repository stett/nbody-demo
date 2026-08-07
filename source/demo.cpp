#include "demo.h"
#include <cmath>
#include <random>
#include "cinder/CinderImGui.h"

using namespace ci;
using namespace ci::gl;

void nbody::Demo::setup()
{
    // Cinder already reports io.DisplaySize in pixels via toPixels(), so DisplayFramebufferScale has
    // to stay at 1. Scaling it by the display's content scale made imgui's GL backend size its
    // framebuffer at DisplaySize * 2, and it derives scissor rects from that height -- so every clip
    // rect landed above the real 1024px framebuffer and the whole UI was scissored away.
    ImGui::Initialize();

    // Prefer the GPU when one is usable. Best effort: a false return just leaves the
    // sim on its default CPU variant, and the combo shows why.
    sim.set_variant(nbody::Variant::GpuBarnesHut);

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
    vbo_particles = gl::Vbo::create(GL_ARRAY_BUFFER, sim.bodies().size() * 3, nullptr, GL_DYNAMIC_DRAW);
    vbo_bounds = gl::Vbo::create(GL_ARRAY_BUFFER, sim.nodes().size() * 7, nullptr, GL_DYNAMIC_DRAW);
    update_gpu_data();

    gl::enableDepthWrite();
    gl::enableDepthRead();

    setup_complete = true;
}

void nbody::Demo::spawn_galaxy(uint32_t num, nbody::util::DiskArgs args)
{
    std::vector<nbody::Body>& bodies = sim.mutable_bodies();
    bodies.resize(bodies.size() + num);
    nbody::util::disk(bodies.end() - num, bodies.end(), args);
}

void nbody::Demo::spawn_cube(uint32_t num, nbody::util::CubeArgs args)
{
    std::vector<nbody::Body>& bodies = sim.mutable_bodies();
    bodies.resize(bodies.size() + num);
    nbody::util::cube(bodies.end() - num, bodies.end(), args);
}

void nbody::Demo::setup_sim_data()
{
    // remove all bodies from the sim
    sim.mutable_bodies().clear();

    // fill the void with evenly spaced stars
    //spawn_cube(target_num_elems, { .size=sim.size });
    //spawn_cube(target_num_elems, { .size=1000 });


    // add a disk galaxy at the origin
    // designators must follow DiskArgs' member order (center, vel, axis)
    spawn_galaxy(target_num_elems, { .center={0,0,0}, .vel={0,0,0}, .axis={0,0,1} });

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
    // Update the CPU buffer for particle data. Read-only: bind const so this per-frame
    // loop never takes mutable access.
    const std::vector<nbody::Body>& bodies = sim.bodies();
    gpu_particle_data.resize(bodies.size() * 8);
    for (size_t i = 0; i < bodies.size(); i++)
    {
        const nbody::Body& body = bodies[i];
        gpu_particle_data[(i * 4) + 0] = (body.pos.x);
        gpu_particle_data[(i * 4) + 1] = (body.pos.y);
        gpu_particle_data[(i * 4) + 2] = (body.pos.z);
        gpu_particle_data[(i * 4) + 3] = (body.radius);
    }

    // Update the GPU buffer
    vbo_particles->bufferData(gpu_particle_data.size() * sizeof(float), gpu_particle_data.data(), GL_DYNAMIC_DRAW);

    // not every simulation variant builds a tree, so tolerate there being none
    const nbody::bh::Tree* bh_tree = sim.tree();
    if (draw_bh_bounds && bh_tree)
    {
        // Update the CPU buffer for tree data
        // Create and populate VBO containing bounds data
        const size_t num_nodes = bh_tree->nodes().size();
        gpu_bounds_data.clear();
        gpu_bounds_data.reserve(7 * num_nodes);
        float max_potential = 0;
        float avg_potential = 0;
        const float num_nodes_inv = 1.f / float(num_nodes);
        for (const nbody::bh::Node& node : bh_tree->nodes())
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
            bh_tree->apply(center, [&potential, &center](const nbody::bh::Node& node) {
                const vec3 delta = vec3(node.com.x, node.com.y, node.com.z) - vec3(center.x, center.y, center.z);
                potential += node.mass / dot(delta, delta);
            });
            max_potential = std::max(max_potential, potential);
            avg_potential += potential * num_nodes_inv;
            gpu_bounds_data.emplace_back(potential);
        }
        const float avg_potential_inv = avg_potential > std::numeric_limits<float>::epsilon() ? 1.f / avg_potential : 0;
        for (size_t i = 0; i < gpu_bounds_data.size(); i += 7)
        {
            float& potential = gpu_bounds_data[i+6];
            potential = std::min(1.f, potential * avg_potential_inv);
        }
        vbo_bounds->bufferData(gpu_bounds_data.size() * sizeof(float), gpu_bounds_data.data(), GL_DYNAMIC_DRAW);
    }
}

void nbody::Demo::resize()
{
#if ! defined(CINDER_MSW)
    // Cinder derives the viewport from glfwGetFramebufferSize() in RendererImplGlfwGl::defaultResize().
    // During startup on macOS that can report the window as still retina-backed, before GLFW settles
    // the NSView for a non-high-density app, so a 1024pt window bakes in a 2048px viewport and nothing
    // re-runs the query afterwards. That put the scene's center in the top-right corner until the first
    // manual resize. Set the viewport from the window size ourselves.
    //
    // Not on MSW. setWindowSize() in setup() dispatches resize() synchronously, and Cinder's
    // WindowImplMsw has its size available before its display is: getSize() is already correct
    // while getContentScale() still reads uninitialised state. toPixels() multiplies the two, so
    // it returns nonsense -- a large negative width is typical -- and the same toPixels() path
    // feeds imgui's DisplaySize from Cinder's NewFrameGuard, so the next NewFrame() trips
    // "Invalid DisplaySize value!" and aborts. That killed better than half of all launches.
    // The MSW renderer sets the viewport correctly on its own, so there is nothing to override.
    const ivec2 size_px = ci::app::toPixels(getWindowSize());
    gl::viewport(0, 0, size_px.x, size_px.y);
#endif

    camera.setPerspective(60, getWindowAspectRatio(), 1, 1e5 );
    gl::setMatrices(camera );
}

void nbody::Demo::update()
{
    // setWindowSize() in setup() dispatches a resize synchronously on some backends,
    // which drives update()/draw() before the shaders and VBOs below exist.
    if (! setup_complete)
        return;

    bool one_tick = false;

    // Update gui
    {
        ImGui::Begin("Settings");
        int app_hz = int(floor(1.f / delta_time));
        ImGui::Text("framerate: %dhz", app_hz);
        if (const nbody::bh::Tree* t = sim.tree())
        {
            const size_t used = t->nodes().size();
            const size_t cap = t->nodes().capacity();
            const int bhtree_percent = cap ? int(100.f * float(used) / float(cap)) : 0;
            ImGui::Text("node capacity: %d (%d%%)", (int)used, bhtree_percent);
        }
        else
        {
            ImGui::Text("node capacity: n/a");
        }
        ImGui::Checkbox("run simulation", &run_simulation);
        int sim_hz = int(ceil(1.f / sim_dt));
        if (ImGui::SliderInt("sim hz", &sim_hz, 1.f, 120.f)) { sim_dt = 1.f / float(sim_hz); }
        if (ImGui::SliderFloat("sim t-scale", &sim_dt_scale, .0f, 1.f)) { }
        if (ImGui::Button("tick simulation")) { one_tick = true; }
        if (ImGui::Button("reset simulation")) { setup_sim_data(); }

        // simulation variant
        {
            const nbody::VariantInfo& current = nbody::Sim::info(sim.variant());
            if (ImGui::BeginCombo("variant", current.name))
            {
                for (const nbody::VariantInfo& info : nbody::Sim::variants())
                {
                    // Latch this before the push. `info` refers into the live variant
                    // table, and a failed switch below marks that same entry
                    // unavailable, so re-reading info.available for the pop would
                    // underflow the style stack.
                    const bool greyed = !info.available;

                    // Grey out unavailable entries by hand rather than with
                    // BeginDisabled, which the ImGui bundled with Cinder predates. They
                    // stay clickable on purpose: set_variant refuses safely and reports
                    // why, so clicking a greyed entry explains itself.
                    if (greyed)
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(.5f, .5f, .5f, 1.f));

                    if (ImGui::Selectable(info.name, info.variant == sim.variant()))
                        variant_error = sim.set_variant(info.variant) ? std::string{} : sim.last_error();

                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("%s", greyed ? info.unavailable_reason.c_str() : info.description);

                    if (greyed)
                        ImGui::PopStyleColor();
                }
                ImGui::EndCombo();
            }
            if (!variant_error.empty())
                ImGui::TextColored(ImVec4(1.f, .4f, .4f, 1.f), "%s", variant_error.c_str());
        }

        bool wrap_space = sim.wrap();
        if (ImGui::Checkbox("wrap space", &wrap_space)) { sim.set_wrap(wrap_space); }

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
    if (! setup_complete)
        return;

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
        gl::drawArrays(GL_POINTS, 0, (GLsizei)sim.nodes().size());
        vbo_bounds->unbind();
        gl::setDefaultShaderVars();
    }

    if (draw_axes)
    {
        gl::color(1, .2, .2, .5);
        gl::drawLine(vec3(-sim.size(), 0, 0), vec3(sim.size(), 0, 0));
        gl::color(.2, 1, .2, .5);
        gl::drawLine(vec3(0, -sim.size(), 0), vec3(0, sim.size(), 0));
        gl::color(.2, .2, 1, .5);
        gl::drawLine(vec3(0, 0, -sim.size()), vec3(0, 0, sim.size()));
    }

    {
        gl::ScopedGlslProg glsl_scope(particle_shader);
        vbo_particles->bind();
        gl::enableVertexAttribArray(0);
        gl::enableVertexAttribArray(1);
        gl::vertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4*sizeof(float), nullptr);
        gl::vertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(3*sizeof(float)));
        gl::drawArrays(GL_POINTS, 0, (GLsizei) sim.bodies().size());
        vbo_particles->unbind();
        gl::setDefaultShaderVars();
    }

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
