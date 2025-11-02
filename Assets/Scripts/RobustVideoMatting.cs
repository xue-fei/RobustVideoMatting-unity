using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

public class RobustVideoMatting : MonoBehaviour
{
    [Header("模型设置")]
    public string modelPath = "rvm_mobilenetv3_fp32.onnx";
    public float downsampleRatio = 0.25f;

    [Header("输入设置")]
    public Texture2D inputTexture;
    public bool useWebcam = false;
    public int webcamWidth = 640;
    public int webcamHeight = 480;

    [Header("输出显示")]
    public Texture2D outputTexture;
    public RawImage rawImage;

    private InferenceSession session;
    private WebCamTexture webCamTexture;

    // 递归状态（隐藏状态）
    private Tensor<float>[] recurrentStates;
    private Tensor<float> downsampleRatioTensor;

    // 输出名称
    private readonly string[] outputNames = { "fgr", "pha", "r1o", "r2o", "r3o", "r4o" };
    private readonly string[] recurrentInputNames = { "r1i", "r2i", "r3i", "r4i" };
    private readonly string[] recurrentOutputNames = { "r1o", "r2o", "r3o", "r4o" };

    // 当前帧结果
    private Texture2D foregroundTexture;
    private Texture2D alphaTexture;
    private Texture2D resultTexture;

    void Start()
    {
        InitializeModel();
        InitializeRecurrentStates();

        if (useWebcam)
        {
            InitializeWebcam();
        }
    }

    void InitializeModel()
    {
        try
        {
            // 加载模型
            var modelFullPath = Path.Combine(Application.streamingAssetsPath, modelPath);

            // 创建会话选项
            var sessionOptions = new SessionOptions();
            var aps = OrtEnv.Instance().GetAvailableProviders();
            foreach (var ap in aps)
            {
                Debug.Log(ap);
            }
            // 设置线程数
            sessionOptions.IntraOpNumThreads = 4;
            sessionOptions.InterOpNumThreads = 4;

            // 设置图优化级别
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            //sessionOptions.AppendExecutionProvider_DML();
            //sessionOptions.AppendExecutionProvider_CPU(); 
            sessionOptions.AppendExecutionProvider_CUDA();

            session = new InferenceSession(modelFullPath, sessionOptions);

            Debug.Log($"Robust Video Matting模型加载成功: {modelPath}");
        }
        catch (Exception e)
        {
            Debug.LogError($"模型初始化失败: {e.Message}");
        }
    }

    void InitializeRecurrentStates()
    {
        try
        {
            // 初始化递归状态 [1, 1, 1, 1]
            recurrentStates = new Tensor<float>[4];
            var zeroData = new float[1] { 0f };
            var shape = new int[] { 1, 1, 1, 1 };

            for (int i = 0; i < 4; i++)
            {
                recurrentStates[i] = new DenseTensor<float>(zeroData, shape);
            }

            // 初始化downsample_ratio
            var ratioData = new float[] { downsampleRatio };
            downsampleRatioTensor = new DenseTensor<float>(ratioData, new int[] { 1 });

            Debug.Log("递归状态初始化完成");
        }
        catch (Exception e)
        {
            Debug.LogError($"递归状态初始化失败: {e.Message}");
        }
    }

    void InitializeWebcam()
    {
        WebCamDevice[] devices = WebCamTexture.devices;
        if (devices.Length > 0)
        {
            webCamTexture = new WebCamTexture(devices[0].name, webcamWidth, webcamHeight, 30);
            webCamTexture.Play();
            Debug.Log($"启动摄像头: {devices[0].name}");
        }
        else
        {
            Debug.LogWarning("未找到摄像头设备");
        }
    }

    void Update()
    {
        if (session == null) return;

        Texture2D sourceTexture = null;

        // 获取输入图像
        if (useWebcam && webCamTexture != null && webCamTexture.isPlaying)
        {
            sourceTexture = WebCamTextureToTexture2D(webCamTexture);
        }
        else if (inputTexture != null)
        {
            sourceTexture = inputTexture;
        }

        if (sourceTexture != null)
        {
            // 处理当前帧
            ProcessFrame(sourceTexture);

            // 显示结果
            if (resultTexture != null)
            {
                rawImage.texture = resultTexture;
            }

            // 清理临时纹理
            if (useWebcam && sourceTexture != null)
            {
                DestroyImmediate(sourceTexture);
            }
        }
    }

    void ProcessFrame(Texture2D sourceTexture)
    {
        try
        {
            // 准备输入张量
            var inputTensor = PrepareInputTensor(sourceTexture);

            // 创建输入列表
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("src", inputTensor),
                NamedOnnxValue.CreateFromTensor("downsample_ratio", downsampleRatioTensor)
            };

            // 添加递归状态输入
            for (int i = 0; i < recurrentInputNames.Length; i++)
            {
                inputs.Add(NamedOnnxValue.CreateFromTensor(recurrentInputNames[i], recurrentStates[i]));
            }

            // 运行推理
            using (var results = session.Run(inputs))
            {
                // 获取输出
                var outputs = ProcessOutputs(results, sourceTexture.width, sourceTexture.height);

                // 更新递归状态
                UpdateRecurrentStates(results);
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"帧处理失败: {e.Message}");
        }
    }

    Tensor<float> PrepareInputTensor(Texture2D texture)
    {
        // 调整尺寸到模型期望的输入大小
        int targetWidth = 512;  // 根据模型调整
        int targetHeight = 512; // 根据模型调整

        var resizedTexture = ResizeTexture(texture, targetWidth, targetHeight);
        Color32[] pixels = resizedTexture.GetPixels32();

        // 创建张量 [1, 3, H, W]
        float[] dataArray = new float[1 * 3 * targetHeight * targetWidth];
        int[] shapeArray = new int[] { 1, 3, targetHeight, targetWidth };

        for (int y = 0; y < targetHeight; y++)
        {
            for (int x = 0; x < targetWidth; x++)
            {
                int index = y * targetWidth + x;
                var pixel = pixels[index];

                // 归一化到 [0, 1]
                int rIndex = 0 * targetHeight * targetWidth + y * targetWidth + x;
                int gIndex = 1 * targetHeight * targetWidth + y * targetWidth + x;
                int bIndex = 2 * targetHeight * targetWidth + y * targetWidth + x;

                dataArray[rIndex] = pixel.r / 255.0f;
                dataArray[gIndex] = pixel.g / 255.0f;
                dataArray[bIndex] = pixel.b / 255.0f;
            }
        }

        // 清理临时纹理
        DestroyImmediate(resizedTexture);

        return new DenseTensor<float>(dataArray, shapeArray, false);
    }

    bool ProcessOutputs(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results, int originalWidth, int originalHeight)
    {
        try
        {
            // 获取前景 (fgr) 和 Alpha (pha) 输出
            var fgrValue = results.FirstOrDefault(r => r.Name == "fgr");
            var phaValue = results.FirstOrDefault(r => r.Name == "pha");

            if (fgrValue == null || phaValue == null)
            {
                Debug.LogError("缺少必要的输出: fgr 或 pha");
                return false;
            }

            var fgrTensor = fgrValue.AsTensor<float>();
            var phaTensor = phaValue.AsTensor<float>();

            // 处理前景纹理
            foregroundTexture = TensorToTexture(fgrTensor, originalWidth, originalHeight, false);

            // 处理Alpha纹理
            alphaTexture = TensorToTexture(phaTensor, originalWidth, originalHeight, true);

            // 创建合成结果
            resultTexture = ComposeResult(foregroundTexture, alphaTexture);

            return true;
        }
        catch (Exception e)
        {
            Debug.LogError($"输出处理失败: {e.Message}");
            return false;
        }
    }

    void UpdateRecurrentStates(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results)
    {
        // 更新递归状态为当前输出的递归状态
        for (int i = 0; i < recurrentOutputNames.Length; i++)
        {
            var stateValue = results.FirstOrDefault(r => r.Name == recurrentOutputNames[i]);
            if (stateValue != null)
            {
                recurrentStates[i] = stateValue.AsTensor<float>();
            }
        }
    }

    Texture2D TensorToTexture(Tensor<float> tensor, int targetWidth, int targetHeight, bool isAlpha)
    {
        var dimensions = tensor.Dimensions.ToArray();
        if (dimensions.Length < 4)
        {
            Debug.LogError($"张量维度不足: {dimensions.Length}");
            return null;
        }

        int tensorHeight = dimensions[2];
        int tensorWidth = dimensions[3];
        int channels = dimensions[1];

        // 创建临时纹理
        var tempTexture = new Texture2D(tensorWidth, tensorHeight,
            isAlpha ? TextureFormat.RFloat : TextureFormat.RGB24, false);

        if (isAlpha && channels == 1)
        {
            // Alpha通道处理
            for (int y = 0; y < tensorHeight; y++)
            {
                for (int x = 0; x < tensorWidth; x++)
                {
                    float alpha = Mathf.Clamp(tensor[0, 0, y, x], 0f, 1f);
                    tempTexture.SetPixel(x, y, new Color(alpha, alpha, alpha));
                }
            }
        }
        else if (!isAlpha && channels == 3)
        {
            // RGB图像处理
            for (int y = 0; y < tensorHeight; y++)
            {
                for (int x = 0; x < tensorWidth; x++)
                {
                    float r = Mathf.Clamp(tensor[0, 0, y, x], 0f, 1f);
                    float g = Mathf.Clamp(tensor[0, 1, y, x], 0f, 1f);
                    float b = Mathf.Clamp(tensor[0, 2, y, x], 0f, 1f);
                    tempTexture.SetPixel(x, y, new Color(r, g, b));
                }
            }
        }

        tempTexture.Apply();

        // 调整到目标尺寸
        var finalTexture = ResizeTexture(tempTexture, targetWidth, targetHeight);
        DestroyImmediate(tempTexture);

        return finalTexture;
    }

    Texture2D ComposeResult(Texture2D foreground, Texture2D alpha)
    {
        if (foreground == null || alpha == null ||
            foreground.width != alpha.width || foreground.height != alpha.height)
        {
            Debug.LogError("前景和Alpha纹理尺寸不匹配");
            return null;
        }

        var result = new Texture2D(foreground.width, foreground.height, TextureFormat.RGBA32, false);
        var fgPixels = foreground.GetPixels();
        var alphaPixels = alpha.GetPixels();

        for (int i = 0; i < fgPixels.Length; i++)
        {
            Color fgColor = fgPixels[i];
            Color alphaColor = alphaPixels[i];

            // 使用Alpha值设置透明度
            fgColor.a = alphaColor.r; // Alpha纹理是灰度图，r通道就是alpha值
            result.SetPixel(i % result.width, i / result.width, fgColor);
        }

        result.Apply();
        return result;
    }

    Texture2D ResizeTexture(Texture2D source, int newWidth, int newHeight)
    {
        var rt = RenderTexture.GetTemporary(newWidth, newHeight);
        Graphics.Blit(source, rt);

        var result = new Texture2D(newWidth, newHeight, TextureFormat.RGBA32, false);
        RenderTexture.active = rt;
        result.ReadPixels(new Rect(0, 0, newWidth, newHeight), 0, 0);
        result.Apply();

        RenderTexture.ReleaseTemporary(rt);
        return result;
    }

    Texture2D WebCamTextureToTexture2D(WebCamTexture webCamTexture)
    {
        Texture2D tex = new Texture2D(webCamTexture.width, webCamTexture.height, TextureFormat.RGBA32, false);
        tex.SetPixels32(webCamTexture.GetPixels32());
        tex.Apply();
        return tex;
    }

    /// <summary>
    /// 重置递归状态（开始新的视频序列时调用）
    /// </summary>
    public void ResetRecurrentStates()
    {
        InitializeRecurrentStates();
        Debug.Log("递归状态已重置");
    }

    /// <summary>
    /// 设置下采样比例
    /// </summary>
    public void SetDownsampleRatio(float ratio)
    {
        downsampleRatio = Mathf.Clamp(ratio, 0.1f, 1.0f);

        // 更新downsample_ratio值
        var ratioData = new float[] { downsampleRatio };
        downsampleRatioTensor = new DenseTensor<float>(ratioData, new int[] { 1 });

        Debug.Log($"下采样比例设置为: {downsampleRatio}");
    }

    /// <summary>
    /// 获取Alpha遮罩纹理
    /// </summary>
    public Texture2D GetAlphaTexture()
    {
        return alphaTexture;
    }

    /// <summary>
    /// 获取前景纹理
    /// </summary>
    public Texture2D GetForegroundTexture()
    {
        return foregroundTexture;
    }

    /// <summary>
    /// 获取结果纹理
    /// </summary>
    public Texture2D GetResultTexture()
    {
        return resultTexture;
    }

    void OnDestroy()
    {
        session?.Dispose();

        if (webCamTexture != null && webCamTexture.isPlaying)
        {
            webCamTexture.Stop();
        }
    }
}