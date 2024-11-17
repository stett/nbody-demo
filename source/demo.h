#pragma once
#include "cinder/app/App.h"
#include "cinder/gl/gl.h"
#include "nbody/sim.h"

using glm::vec2;
using glm::vec3;
using glm::ivec2;
using glm::ivec3;
using glm::mat4;
using glm::quat;
using ci::app::MouseEvent;
using ci::CameraPersp;
using ci::gl::GlslProgRef;
using ci::gl::VboRef;

namespace nbody {
    class Demo : public ci::app::App {
    public:
        void setup() override;
        void resize() override;
        void update() override;
        void draw() override;
        void mouseMove(MouseEvent event) override;
        void mouseDrag(MouseEvent event) override;
        void mouseWheel(MouseEvent event) override;

    private:

        void setup_sim_data();
        void setup_acceleration_structure();
        void update_gpu_data();
        void update_selected_body();

        // helpers
        vec3 homogeneous_to_world(const vec3 &homo) const;
        void mouse_ray(vec3 &out_ray_origin, vec3 &out_ray_direction) const;

        // nbody sim;
        nbody::Sim sim;

        // time
        double time = 0;
        float delta_time = 0;
        float sim_dt = 1.f / 60.f;
        float sim_dt_accum = 0;
        float sim_dt_scale = 1.f;

        // shaders
        GlslProgRef bounds_shader;
        GlslProgRef particle_shader;

        // gpu data caches
        std::vector<float> gpu_particle_data;
        VboRef vbo_particles;
        std::vector<float> gpu_bounds_data;
        VboRef vbo_bounds;

        // settings
        bool run_simulation = false;
        bool draw_bh_bounds = false;
        bool draw_axes = false;
        bool draw_selection = false;
        bool draw_collisions = false;
        int32_t selected_elem = 0;
        size_t target_num_elems = 4096;

        // camera
        CameraPersp camera;
        vec3 cam_focus = glm::vec3(0);
        vec3 cam_focus_target = glm::vec3(0);
        vec2 cam_angles = glm::vec2(M_PI * .5, 0);
        vec2 cam_target_angles = glm::vec2(M_PI * .5, 0);
        float cam_dist = sim.size;
        float cam_target_dist = sim.size;

        // mouse
        ivec2 mouse_pos;
        ivec2 mouse_delta;
    };
}
