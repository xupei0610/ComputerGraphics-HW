//CSCI 5607 OpenGL Tutorial (HW 1/2)
//5 - Model Load

#include "glad/glad.h"  //Include order can matter here
#ifdef __APPLE__
#include <SDL2/SDL.h>
 #include <SDL2/SDL_opengl.h>
#else
#include <SDL.h>
#include <SDL_opengl.h>
#endif
#include <cstdio>

//#include <vector>
//#include <algorithm>


#define GLM_FORCE_RADIANS
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

bool saveOutput = false;
float timePast = 0;

// Shader sources
const GLchar* vertexSource =
        "#version 150 core\n"
                "in vec3 position;"
//  "in vec3 inColor;"
                "const vec3 inColor = vec3(0.f,0.7f,0.f);"
                "in vec3 inNormal;"
                "uniform int shading_mode;"
                "flat out int shadingMode;"
                "out vec3 refColor;"
                "out vec3 Color;"
                "out vec3 normal;"
                "out vec3 pos;"
                "uniform mat4 model;"
                "uniform mat4 view;"
                "uniform mat4 proj;"
                "const vec3 lightPos = vec3(1,1,1);"
                "void main() {"
                "   Color = inColor;"
                "   gl_Position = proj * view * model * vec4(position,1.0);"
                "   pos = (model * vec4(position,1.0)).xyz;"
                "   vec4 norm4 = transpose(inverse(model)) * vec4(inNormal,0.0);"  //Just model, than noramlize normal
                "   normal = normalize(norm4.xyz);"
                "   shadingMode = shading_mode;"
                "   if (shadingMode == 3 || shadingMode == 4) {"
                "       vec3 lightDir = normalize(lightPos-pos);"
                "       vec3 diffuseC = Color*max(dot(lightDir, normal),0.0);"
                "       vec3 viewDir = normalize(-pos);" //We know the eye is at (0,0)!
                "       float spec;"
                "       if (shadingMode == 4) {"
                "           vec3 h = normalize(lightDir - viewDir);"
                "           spec = dot(h, normal);"
                "       } else {"
                "           vec3 r= reflect(lightDir, normal);"
                "           spec = dot(r, viewDir);"
                "       }"
                "       spec = max(spec, 0.0);"
                "       if (dot(lightDir,normal) <= 0.0)spec = 0;"
                "       vec3 specC = vec3(.8, .8, .8)*pow(spec,32);"
                "       refColor = diffuseC+specC;"
                "   }"
                "}";

const GLchar* fragmentSource =
        "#version 150 core\n"
                "in vec3 Color;"
                "in vec3 normal;"
                "in vec3 pos;"
                "in vec3 refColor;"
                "flat in int shadingMode;"
                "out vec4 outColor;"
                "const vec3 lightPos = vec3(1,1,1);"
                "const float ambient = .3;"
                "void main() {"
                "   vec3 ambC = Color*ambient;"
                "   if (shadingMode == 3 || shadingMode == 4) {"
                "   "
                "       outColor = vec4(ambC+refColor, 1.0);"
                "       return;"
                "   }"
                "   vec3 lightDir = normalize(lightPos-pos);"
                "   vec3 diffuseC = Color*max(dot(lightDir, normal),0.0);"
                "   vec3 viewDir = normalize(-pos);" //We know the eye is at (0,0)!
                "   float spec;"
                "   if (shadingMode == 2) {"
                "       vec3 h = normalize(lightDir - viewDir);"
                "       spec = dot(h, normal);"
                "   } else {"
                "       vec3 r= reflect(lightDir, normal);"
                "       spec = dot(r, viewDir);"
                "   }"
                "   spec = max(spec, 0.0);"
                "   if (dot(lightDir,normal) <= 0.0)spec = 0;"
                "       vec3 specC = vec3(.8, .8, .8)*pow(spec,32);"
                "   outColor = vec4(diffuseC+ambC+specC, 1.0);"
                "}";

GLint shading_mode = 1;

bool fullscreen = false;
int screen_width = 800;
int screen_height = 600;

void Win2PPM(int width, int height);

int main(int argc, char *argv[]){
    SDL_Init(SDL_INIT_VIDEO);  //Initialize Graphics (for OpenGL)

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

    ifstream modelFile;
    if (argc > 1)
    {
        modelFile.open(argv[1]);
    }
    if (argc < 2 || !modelFile.is_open())
    {
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR,
                                 "Load Model Error",
                                 "Failed to Open Model File.",
                                 NULL);
        printf("Failed to Open Model File\n");
    }
    int numLines = 0;
    modelFile >> numLines;
    float* modelData = new float[numLines];
    for (int i = 0; i < numLines; i++){
        modelFile >> modelData[i];
    }
    printf("Mode line count: %d\n",numLines);
    float numTris = numLines/8;

    // Add by Pei Xu
    // extract model into a scn file used by ray tracing
//    std::vector<std::vector<float> > vmap;
//    std::vector<std::vector<float> > nmap;
//    std::vector<std::vector<int> > tri;
//    float xmin = 1, ymin = 1, zmin = 1;
//    float xmax =-1, ymax =-1, zmax =-1;
//    int index = 0;
//    for (auto i = 0; i < numLines/8; i+=3)
//    {
//        auto src = i*8;
//        std::vector<float> v1{modelData[src],modelData[src+1],modelData[src+2]};
//        std::vector<float> n1{modelData[src+5],modelData[src+6],modelData[src+7]};
//        std::vector<float> v2{modelData[src+8],modelData[src+1+8],modelData[src+2+8]};
//        std::vector<float> n2{modelData[src+5+8],modelData[src+6+8],modelData[src+7+8]};
//        std::vector<float> v3{modelData[src+8+8],modelData[src+1+8+8],modelData[src+2+8+8]};
//        std::vector<float> n3{modelData[src+5+8+8],modelData[src+6+8+8],modelData[src+7+8+8]};
//
//        if (std::find(vmap.begin(), vmap.end(), v1) == vmap.end())
//            vmap.push_back(v1);
//        if (std::find(nmap.begin(), nmap.end(), n1) == nmap.end())
//            nmap.push_back(n1);
//        if (std::find(vmap.begin(), vmap.end(), v2) == vmap.end())
//            vmap.push_back(v2);
//        if (std::find(nmap.begin(), nmap.end(), n2) == nmap.end())
//            nmap.push_back(n2);
//        if (std::find(vmap.begin(), vmap.end(), v3) == vmap.end())
//            vmap.push_back(v3);
//        if (std::find(nmap.begin(), nmap.end(), n3) == nmap.end())
//            nmap.push_back(n3);
//
//        if (v1[0] > xmax) xmax = v1[0];
//        if (v1[0] < xmin) xmin = v1[0];
//        if (v1[1] > ymax) ymax = v1[1];
//        if (v1[1] < ymin) ymin = v1[1];
//        if (v1[2] > zmax) zmax = v1[2];
//        if (v1[2] < zmin) zmin = v1[2];
//
//        if (v2[0] > xmax) xmax = v2[0];
//        if (v2[0] < xmin) xmin = v2[0];
//        if (v2[1] > ymax) ymax = v2[1];
//        if (v2[1] < ymin) ymin = v2[1];
//        if (v2[2] > zmax) zmax = v2[2];
//        if (v2[2] < zmin) zmin = v2[2];
//
//
//        if (v3[0] > xmax) xmax = v3[0];
//        if (v3[0] < xmin) xmin = v3[0];
//        if (v3[1] > ymax) ymax = v3[1];
//        if (v3[1] < ymin) ymin = v3[1];
//        if (v3[2] > zmax) zmax = v3[2];
//        if (v3[2] < zmin) zmin = v3[2];
//
//        tri.push_back(std::vector<int>{std::distance(vmap.begin(), std::find(vmap.begin(), vmap.end(), v1)),
//                                       std::distance(vmap.begin(), std::find(vmap.begin(), vmap.end(), v2)),
//                                       std::distance(vmap.begin(), std::find(vmap.begin(), vmap.end(), v3)),
//                                       std::distance(nmap.begin(), std::find(nmap.begin(), nmap.end(), n1)),
//                                       std::distance(nmap.begin(), std::find(nmap.begin(), nmap.end(), n2)),
//                                       std::distance(nmap.begin(), std::find(nmap.begin(), nmap.end(), n3))});
//        std::cout << i << " " << v1[0] << " " << v1[1] << " " << v1[2] << std::endl;
//    }
//
//    std::ofstream fs;
//    fs.open("info.scn");
//    for (const auto & v : vmap)
//    {
//        fs << "vertex " << v[0] << " " << v[1] << " " << v[2] << "\n";
//    }
//    fs << "\n";
//    for (const auto & n : nmap)
//    {
//        fs << "normal " << n[0] << " " << n[1] << " " << n[2] << "\n";
//    }
//    fs << "\n";
//    for (const auto & t : tri)
//    {
//        fs << "normal_triangle " << t[0] << " " << t[1] << " " << t[2] << " " << t[3] << " " << t[4] << " " << t[5] << "\n";
//    }
//    fs << std::endl;
//    fs.close();
//    std::cout << "(" << xmin << ", " << ymin << ", " << zmin << ")"
//              << "(" << xmax << ", " << ymax << ", " << zmax << ")" << std::endl;
    // end Add by Pei Xu

    //Load the vertex Shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);

    //Let's double check the shader compiled
    GLint status;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &status);
    if (!status){
        char buffer[512];
        glGetShaderInfoLog(vertexShader, 512, NULL, buffer);
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR,
                                 "Compilation Error",
                                 "Failed to Compile: Check Consol Output.",
                                 NULL);
        printf("Vertex Shader Compile Failed. Info:\n\n%s\n",buffer);
    }

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);

    //Double check the shader compiled
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &status);
    if (!status){
        char buffer[512];
        glGetShaderInfoLog(fragmentShader, 512, NULL, buffer);
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR,
                                 "Compilation Error",
                                 "Failed to Compile: Check Consol Output.",
                                 NULL);
        printf("Fragment Shader Compile Failed. Info:\n\n%s\n",buffer);
    }

    //Join the vertex and fragment shaders together into one program
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glBindFragDataLocation(shaderProgram, 0, "outColor"); // set output
    glLinkProgram(shaderProgram); //run the linker

    // Add by Pei Xu
    // Link error check
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &status);
    if (status == GL_FALSE)
    {
//            glGetProgramiv(shaderProgram, GL_INFO_LOG_LENGTH, &status);
//            std::vector<GLchar> err_msg(status);
        char buffer[512];
        glGetProgramInfoLog(shaderProgram, 512, NULL, buffer);
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR,
                                 "Link Error",
                                 "Failed to Link Program: Check Console Output.",
                                 NULL);
        printf("Shader Program Link Failed. Info:\n\n%s\n",buffer);
    }
    // end Add by Pei Xu


    glUseProgram(shaderProgram); //Set the active shader (only one can be used at a time)


    //Build a Vertex Array Object. This stores the VBO and attribute mappings in one object
    GLuint vao;
    glGenVertexArrays(1, &vao); //Create a VAO
    glBindVertexArray(vao); //Bind the above created VAO to the current context

    //Allocate memory on the graphics card to store geometry (vertex buffer object)
    GLuint vbo[1];
    glGenBuffers(1, vbo);  //Create 1 buffer called vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); //Set the vbo as the active array buffer (Only one buffer can be active at a time)
    glBufferData(GL_ARRAY_BUFFER, numLines*sizeof(float), modelData, GL_STATIC_DRAW); //upload vertices to vbo
    //GL_STATIC_DRAW means we won't change the geometry, GL_DYNAMIC_DRAW = geometry changes infrequently
    //GL_STREAM_DRAW = geom. changes frequently.  This effects which types of GPU memory is used

    //Tell OpenGL how to set fragment shader input
    GLint posAttrib = glGetAttribLocation(shaderProgram, "position");
    glVertexAttribPointer(posAttrib, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), 0);
    //Attribute, vals/attrib., type, normalized?, stride, offset
    //Binds to VBO current GL_ARRAY_BUFFER
    glEnableVertexAttribArray(posAttrib);

//    GLint colAttrib = glGetAttribLocation(shaderProgram, "inColor");
//    glVertexAttribPointer(colAttrib, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(3*sizeof(float)));
//    glEnableVertexAttribArray(colAttrib);

    GLint normAttrib = glGetAttribLocation(shaderProgram, "inNormal");
    glVertexAttribPointer(normAttrib, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(5*sizeof(float)));
    glEnableVertexAttribArray(normAttrib);

    glBindVertexArray(0); //Unbind the VAO

    //Maybe we need a second VAO, e.g., some of our models are stored in a second format
    //GLuint vao2;
    //glGenVertexArrays(1, &vao2); //Create the VAO
    //glBindVertexArray(vao2); //Bind the above created VAO to the current context
    //  Creat VBOs ...
    //  Set-up attributes ...
    //glBindVertexArray(0); //Unbind the VAO


    glEnable(GL_DEPTH_TEST);

    //Event Loop (Loop forever processing each event as fast as possible)
    SDL_Event windowEvent;

    glm::mat4 model;

    while (true){

        if (SDL_PollEvent(&windowEvent)){
            if (windowEvent.type == SDL_QUIT) break;
            //List of keycodes: https://wiki.libsdl.org/SDL_Keycode - You can catch many special keys
            //Scancode referes to a keyboard position, keycode referes to the letter (e.g., EU keyboards)
            if (windowEvent.type == SDL_KEYUP && windowEvent.key.keysym.sym == SDLK_ESCAPE)
                break; //Exit event loop
            if (windowEvent.type == SDL_KEYUP && windowEvent.key.keysym.sym == SDLK_f) //If "f" is pressed
            {
                fullscreen = !fullscreen;
                SDL_SetWindowFullscreen(window,
                                        fullscreen ? SDL_WINDOW_FULLSCREEN : 0); //Toggle fullscreen
            }
                // Add by Pei Xu
            else if (windowEvent.type == SDL_KEYDOWN)
            {
                if (windowEvent.key.keysym.sym == SDLK_r)
                    model = glm::mat4();
                else if (windowEvent.key.keysym.sym == SDLK_w || windowEvent.key.keysym.sym == SDLK_UP)
                    model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.1f));
                else if (windowEvent.key.keysym.sym == SDLK_s || windowEvent.key.keysym.sym == SDLK_DOWN)
                    model = glm::translate(model, glm::vec3(0.0f, 0.0f, -0.1f));
                else if (windowEvent.key.keysym.sym == SDLK_a || windowEvent.key.keysym.sym == SDLK_LEFT)
                    model = glm::translate(model, glm::vec3(0.0f, -0.1f, 0.0f));
                else if (windowEvent.key.keysym.sym == SDLK_d || windowEvent.key.keysym.sym == SDLK_RIGHT)
                    model = glm::translate(model, glm::vec3(0.0f, 0.1f, 0.0f));
                else if (windowEvent.key.keysym.sym == SDLK_z || windowEvent.key.keysym.sym == SDLK_PLUS)
                    model = glm::translate(model, glm::vec3(0.1f, 0.0f, 0.0f));
                else if (windowEvent.key.keysym.sym == SDLK_x || windowEvent.key.keysym.sym == SDLK_MINUS)
                    model = glm::translate(model, glm::vec3(-0.1f, 0.0f, 0.0f));
                else if (windowEvent.key.keysym.sym == SDLK_q || windowEvent.key.keysym.sym == SDLK_LEFT)
                    model = glm::rotate(model, -3.14f/90, glm::vec3(0.0f, 0.0f, 1.0f));
                else if (windowEvent.key.keysym.sym == SDLK_e || windowEvent.key.keysym.sym == SDLK_RIGHT)
                    model = glm::rotate(model, -3.14f/90, glm::vec3(1.0f, 0.0f, 0.0f));
                else if (windowEvent.key.keysym.sym == SDLK_c)
                    model = glm::rotate(model, -3.14f/90, glm::vec3(0.0f, 1.0f, 0.0f));
            }
            else if (windowEvent.type == SDL_KEYUP)
            {
                GLint old = shading_mode;
                if (windowEvent.key.keysym.sym == SDLK_1)
                    // Phong
                    shading_mode = 1;
                else if (windowEvent.key.keysym.sym == SDLK_2)
                    // Blinn-Phong
                    shading_mode = 2;
                else if (windowEvent.key.keysym.sym == SDLK_3)
                    // Gouraud Phong
                    shading_mode = 3;
                else if (windowEvent.key.keysym.sym == SDLK_4)
                    // Gouraud Blinn-Phong
                    shading_mode = 4;
                if (old != shading_mode)
                {
                    std::cout << "Shading Mode set to: "
                              << (shading_mode == 1 ? "Phong" :
                                  shading_mode == 2 ? "Blinn-Phong" :
                                  shading_mode == 3 ? "Gouraud with Phong" :
                                                      "Gouraud with Blinn-Phong")
                              << std::endl;
                }
            }
            // end Add by Pei
        }


        GLint mode_loc = glGetUniformLocation(shaderProgram, "shading_mode");
        glUniform1iv(mode_loc, 1, &shading_mode);


        // Clear the screen to default color
        glClearColor(.2f, 0.4f, 0.8f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//        if (!saveOutput) timePast = SDL_GetTicks()/1000.f;
//        if (saveOutput) timePast += .07; //Fix framerate at 14 FPS
//        model = glm::rotate(model,timePast * 3.14f/2,glm::vec3(0.0f, 1.0f, 1.0f));
//        model = glm::rotate(model,timePast * 3.14f/4,glm::vec3(1.0f, 0.0f, 0.0f));
        GLint uniModel = glGetUniformLocation(shaderProgram, "model");
        glUniformMatrix4fv(uniModel, 1, GL_FALSE, glm::value_ptr(model));

        glm::mat4 view = glm::lookAt(
                glm::vec3(3.f, 0.f, 0.f),  //Cam Position
                glm::vec3(0.0f, 0.0f, 0.0f),  //Look at point
                glm::vec3(0.0f, 0.0f, 1.0f)); //Up
        GLint uniView = glGetUniformLocation(shaderProgram, "view");
        glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view));

        glm::mat4 proj = glm::perspective(3.14f/4, aspect, 1.0f, 10.0f); //FOV, aspect, near, far
        GLint uniProj = glGetUniformLocation(shaderProgram, "proj");
        glUniformMatrix4fv(uniProj, 1, GL_FALSE, glm::value_ptr(proj));

        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, numTris); //(Primitives, Which VBO, Number of vertices)
        if (saveOutput) Win2PPM(screen_width,screen_height);


        SDL_GL_SwapWindow(window); //Double buffering
    }


    //Clean Up
    glDeleteProgram(shaderProgram);
    glDeleteShader(fragmentShader);
    glDeleteShader(vertexShader);

    glDeleteBuffers(1, vbo);
    glDeleteVertexArrays(1, &vao);

    SDL_GL_DeleteContext(context);
    SDL_Quit();
    return 0;
}


//Write out PPM image from screen
void Win2PPM(int width, int height){
    char outdir[10] = "out/"; //Must be defined!
    int i,j;
    FILE* fptr;
    static int counter = 0;
    char fname[32];
    unsigned char *image;

    /* Allocate our buffer for the image */
    image = (unsigned char *)malloc(3*width*height*sizeof(char));
    if (image == NULL) {
        fprintf(stderr,"ERROR: Failed to allocate memory for image\n");
    }

    /* Open the file */
    sprintf(fname,"%simage_%04d.ppm",outdir,counter);
    if ((fptr = fopen(fname,"w")) == NULL) {
        fprintf(stderr,"ERROR: Failed to open file to write image\n");
    }

    /* Copy the image into our buffer */
    glReadBuffer(GL_BACK);
    glReadPixels(0,0,width,height,GL_RGB,GL_UNSIGNED_BYTE,image);

    /* Write the PPM file */
    fprintf(fptr,"P6\n%d %d\n255\n",width,height);
    for (j=height-1;j>=0;j--) {
        for (i=0;i<width;i++) {
            fputc(image[3*j*width+3*i+0],fptr);
            fputc(image[3*j*width+3*i+1],fptr);
            fputc(image[3*j*width+3*i+2],fptr);
        }
    }

    free(image);
    fclose(fptr);
    counter++;
}
