#include "demo.h"
#include "cinder/app/RendererGl.h"

namespace
{
    void prepareSettings(ci::app::App::Settings *settings)
    {
        settings->setHighDensityDisplayEnabled(true);
    }
}

using ci::app::RendererGl;
CINDER_APP(nbody::Demo, RendererGl(RendererGl::Options().msaa(16)), prepareSettings)
