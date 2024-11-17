#include "demo.h"
#include "cinder/app/RendererGl.h"

using ci::app::RendererGl;
CINDER_APP(nbody::Demo, RendererGl(RendererGl::Options().msaa(16)))
