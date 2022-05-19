#ifdef GL_ES
precision mediump float;
#endif

// Phong related variables
uniform sampler2D uSampler;
uniform vec3 uKd;
uniform vec3 uKs;
uniform vec3 uLightPos;
uniform vec3 uCameraPos;
uniform vec3 uLightIntensity;

varying highp vec2 vTextureCoord;
varying highp vec3 vFragPos;
varying highp vec3 vNormal;

// Shadow map related variables
#define NUM_SAMPLES 20
#define BLOCKER_SEARCH_NUM_SAMPLES NUM_SAMPLES
#define PCF_NUM_SAMPLES NUM_SAMPLES
#define NUM_RINGS 10

#define EPS 1e-3
#define PI 3.141592653589793
#define PI2 6.283185307179586

uniform sampler2D uShadowMap;

varying vec4 vPositionFromLight;

highp float rand_1to1(highp float x ) { 
  // -1 -1
  return fract(sin(x)*10000.0);
}

highp float rand_2to1(vec2 uv ) { 
  // 0 - 1
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract(sin(sn) * c);
}

float unpack(vec4 rgbaDepth) {
    //const vec4 bitShift = vec4(1.0, 1.0/256.0, 1.0/(256.0*256.0), 1.0/(256.0*256.0*256.0));
    const vec4 bitShift = vec4(1.0, 1.0/255.0, 1.0/(255.0*255.0), 1.0/(255.0*255.0*255.0));

    return dot(rgbaDepth, bitShift);
}

vec2 poissonDisk[NUM_SAMPLES];

void poissonDiskSamples( const in vec2 randomSeed ) {

  float ANGLE_STEP = PI2 * float( NUM_RINGS ) / float( NUM_SAMPLES );
  float INV_NUM_SAMPLES = 1.0 / float( NUM_SAMPLES );

  float angle = rand_2to1( randomSeed ) * PI2;
  float radius = INV_NUM_SAMPLES;
  float radiusStep = radius;

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( cos( angle ), sin( angle ) ) * pow( radius, 0.75 );
    radius += radiusStep;
    angle += ANGLE_STEP;
  }
}

void uniformDiskSamples( const in vec2 randomSeed ) {

  float randNum = rand_2to1(randomSeed);
  float sampleX = rand_1to1( randNum ) ;
  float sampleY = rand_1to1( sampleX ) ;

  float angle = sampleX * PI2;
  float radius = sqrt(sampleY);

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( radius * cos(angle) , radius * sin(angle)  );

    sampleX = rand_1to1( sampleY ) ;
    sampleY = rand_1to1( sampleX ) ;

    angle = sampleX * PI2;
    radius = sqrt(sampleY);
  }
}

float findBlocker( sampler2D shadowMap,  vec2 uv, float zReceiver ) {
	return 1.0;
}

float PCF(sampler2D shadowMap, vec4 coords) {
  return 1.0;
}

float PCSS(sampler2D shadowMap, vec4 coords){

  // STEP 1: avgblocker depth

  // STEP 2: penumbra size

  // STEP 3: filtering
  
  return 1.0;

}
float Bias(){
 //解决shadow bias 因为shadow map的精度有限，当要渲染的fragment在light space中距Light很远的时候，就会有多个附近的fragement会samper shadow map中同一个texel,但是即使这些fragment在camera view space中的深度值z随xy变化是值变化是很大的，
  //但他们在light space 中的z值(shadow map中的值)却没变或变化很小，这是因为shadow map分辨率低，采样率低导致精度低，不能准确的记录这些细微的变化
 
  // calculate bias (based on depth map resolution and slope)  vec3 lightDir = normalize(uLightPos);
  vec3 lightDir = normalize(uLightPos);
  vec3 normal = normalize(vNormal);
  float bias = max(0.005 * (1.0 - dot(normal, lightDir)), 0.005);
  return bias;
}

//Task 2
// Camera Pass
//查询当前着色点在 ShadowMap 上记录的深度值，并与转换到 light space 的深度值比较后返回 visibility 项
//查询坐标需要先转换到 NDC 标准空间 [0,1]
float useShadowMap(sampler2D shadowMap, vec4 shadowCoord){
  float bias = Bias();  // 1e-5
  vec4 depthpack = texture2D(shadowMap, shadowCoord.xy);
  float depthUnpack = unpack(depthpack);//unpack后已经在[0,1]
  //如果shadow map的depth < lightsource->shading point 的distance = shadow
  if(depthUnpack > shadowCoord.z - bias)
      return 1.0;
  return 0.0;


  // float depthOnShadowMap = unpack(texture2D(shadowMap, shadowCoord.xy));
  // if (abs(depthOnShadowMap) < 1e-5) depthOnShadowMap = 1.0;
  // float depth = shadowCoord.z;
  
  // float vis = step(depth - EPS, depthOnShadowMap);
  // return vis;
}

vec3 blinnPhong() {
  vec3 color = texture2D(uSampler, vTextureCoord).rgb;
  color = pow(color, vec3(2.2));

  vec3 ambient = 0.05 * color;

  vec3 lightDir = normalize(uLightPos);
  vec3 normal = normalize(vNormal);
  float diff = max(dot(lightDir, normal), 0.0);
  vec3 light_atten_coff =
      uLightIntensity / pow(length(uLightPos - vFragPos), 2.0);
  vec3 diffuse = diff * light_atten_coff * color;

  vec3 viewDir = normalize(uCameraPos - vFragPos);
  vec3 halfDir = normalize((lightDir + viewDir));
  float spec = pow(max(dot(halfDir, normal), 0.0), 32.0);
  vec3 specular = uKs * light_atten_coff * spec;

  vec3 radiance = (ambient + diffuse + specular);
  vec3 phongColor = pow(radiance, vec3(1.0 / 2.2));
  return phongColor;
}

void main(void) {
  //version 1 
  float visibility;
  vec3 phongColor = blinnPhong();


  //shadowCoord 需要normalize 到 NDC
  //1 vPositionFromLight 需要从齐次转回来（-1，1），然后转为(0,1)
  vec3 shadowCoordNDC = (vPositionFromLight.xyz / vPositionFromLight.w + 1.0) / 2.0;

  
  visibility = useShadowMap(uShadowMap, vec4(shadowCoordNDC, 1.0));
  // //visibility = PCF(uShadowMap, vec4(shadowCoord, 1.0));
  // //visibility = PCSS(uShadowMap, vec4(shadowCoord, 1.0));

  
  //0: 无颜色 黑
  gl_FragColor = vec4(phongColor * visibility, 1.0);



  //origin
  //gl_FragColor = vec4(phongColor, 1.0);


}