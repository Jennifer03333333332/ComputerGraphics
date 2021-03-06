class DirectionalLight {

    constructor(lightIntensity, lightColor, lightPos, focalPoint, lightUp, hasShadowMap, gl) {
        this.mesh = Mesh.cube(setTransform(0, 0, 0, 0.2, 0.2, 0.2, 0));
        this.mat = new EmissiveMaterial(lightIntensity, lightColor);
        this.lightPos = lightPos;
        this.focalPoint = focalPoint;
        this.lightUp = lightUp

        this.hasShadowMap = hasShadowMap;
        this.fbo = new FBO(gl);
        if (!this.fbo) {
            console.log("无法设置帧缓冲区对象");
            return;
        }
    }
    //using orthogonal projection
    //translate: this mesh
    CalcLightMVP(translate, scale) {
        let lightMVP = mat4.create();
        let modelMatrix = mat4.create();
        let viewMatrix = mat4.create();
        let projectionMatrix = mat4.create();
        //consider light as camera
        // Model transform
        // 这些都是来自gl-matrix的函数，第一个参数out：存放结果
        //mat4.identity(modelMatrix);
		mat4.translate(modelMatrix, modelMatrix, translate);//this.mesh.transform.
		mat4.scale(modelMatrix, modelMatrix, scale);
        // View transform. (out, eye, center, up). View = Rview*Tview
        mat4.lookAt(viewMatrix, this.lightPos, this.focalPoint, this.lightUp);
        // Projection transform. ortho(out, left, right, bottom, top, near, far)
        // 参数为被clip出来的区域，需要把目标物体框住
        //mat4.ortho(projectionMatrix,-500,500,-500,500,0.1,1000);
        //mat4.ortho(projectionMatrix, -500.0, 500.0, -500.0, 500.0, 0.1, 500);
        //改变参数后锯齿少了很多 ？
        mat4.ortho(projectionMatrix, -120.0, 120.0, -120.0, 120.0, 1.0, 500);
        //顺序： projectionMatrix * viewMatrix
        mat4.multiply(lightMVP, projectionMatrix, viewMatrix);
        mat4.multiply(lightMVP, lightMVP, modelMatrix);

        return lightMVP;
    }
}
