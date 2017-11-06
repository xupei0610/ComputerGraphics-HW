//CSCI 5607 OpenGL Tutorial (HW 1/2)
//1 - Blank Screen

#include "glad/glad.h"  //Include order can matter here
#ifdef __APPLE__
 #include <SDL2/SDL.h>
 #include <SDL2/SDL_opengl.h>
#else
 #include <SDL.h>
 #include <SDL_opengl.h>
#endif
#include <cstdio>

bool fullscreen = false;
int screen_width = 800;
int screen_height = 600;

int main(int argc, char *argv[]){
   SDL_Init(SDL_INIT_VIDEO);  //Initialize Graphics (for OpenGL)

   	//Print the version of SDL we are using (should be 2.0.5 or higher)
	SDL_version compiled; SDL_version linked;
	SDL_VERSION(&compiled); SDL_GetVersion(&linked);
    printf("\nCompiled against SDL version %d.%d.%d ...\n", compiled.major, compiled.minor, compiled.patch);
	printf("Linking against SDL version %d.%d.%d.\n", linked.major, linked.minor, linked.patch);
    
    //Ask SDL to get a recent version of OpenGL (3.2 or greater)
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
	
	//Create a window (offsetx, offsety, width, height, flags)
	SDL_Window* window = SDL_CreateWindow("My OpenGL Program", 100, 100, screen_width, screen_height, SDL_WINDOW_OPENGL);
	float aspect = screen_width/(float)screen_height; //aspect ratio (needs to be updated if the window is resized)
	
	//The above window cannot be resized which makes some code slightly easier.
	//Below show how to make a full screen window or allow resizing
	//SDL_Window* window = SDL_CreateWindow("My OpenGL Program", 0, 0, screen_width, screen_height, SDL_WINDOW_FULLSCREEN|SDL_WINDOW_OPENGL);
	//SDL_Window* window = SDL_CreateWindow("My OpenGL Program", 100, 100, screen_width, screen_height, SDL_WINDOW_RESIZABLE|SDL_WINDOW_OPENGL);
	//SDL_Window* window = SDL_CreateWindow("My OpenGL Program",SDL_WINDOWPOS_UNDEFINED,SDL_WINDOWPOS_UNDEFINED,0,0,SDL_WINDOW_FULLSCREEN_DESKTOP|SDL_WINDOW_OPENGL); //Boarderless window "fake" full screen
	
	//Create a context to draw in
	SDL_GLContext context = SDL_GL_CreateContext(window);
    
	//OpenGL functions using glad library
    if (gladLoadGLLoader(SDL_GL_GetProcAddress)){
       printf("\nOpenGL loaded\n");
       printf("Vendor:   %s\n", glGetString(GL_VENDOR));
       printf("Renderer: %s\n", glGetString(GL_RENDERER));
       printf("Version:  %s\n\n", glGetString(GL_VERSION));
    }
    else {
        printf("ERROR: Failed to initialize OpenGL context.\n");
        return -1;
    }
	
	GLuint vertexBuffer;
	glGenBuffers(1, &vertexBuffer);
	printf("Gereated Vertex Buffer: #%u\n", vertexBuffer);
	
	
	//Event Loop (Loop forever processing each event as fast as possible
	SDL_Event windowEvent;
	bool quit = false;
	while (!quit){
      while (SDL_PollEvent(&windowEvent)){
        if (windowEvent.type == SDL_QUIT) quit=true; //Exit event loop
        if (windowEvent.type == SDL_KEYUP && windowEvent.key.keysym.sym == SDLK_ESCAPE) 
          quit=true; //Exit event loop
        if (windowEvent.type == SDL_KEYUP && windowEvent.key.keysym.sym == SDLK_f) //If "f" is pressed
          fullscreen = !fullscreen;
          SDL_SetWindowFullscreen(window, fullscreen ? SDL_WINDOW_FULLSCREEN : 0); //Toggle fullscreen 
      }
      
      glClearColor(.2f, 0.4f, 0.8f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

     SDL_GL_SwapWindow(window);
	}

	//Clean Up
	SDL_GL_DeleteContext(context);
	SDL_Quit();
	return 0;
}