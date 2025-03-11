import { useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleFileSelect = (event) => {
    const file = event.target.files[0]
    if (file) {
      setSelectedFile(file)
      setPreview(URL.createObjectURL(file))
      setPrediction(null)
      setError(null)
    }
  }

  const handleSubmit = async () => {
    if (!selectedFile) return

    setLoading(true)
    setError(null)
    
    const formData = new FormData()
    formData.append('file', selectedFile)
    
    try {
      const response = await axios.post('http://localhost:8000/predict', 
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      )
      setPrediction(response.data)
    } catch (err) {
      console.error("Error submitting image:", err)
      setError("Failed to get prediction. Please try again.")
    } finally {
      setLoading(false)
    }
  }

  const confidencePercent = prediction ? Math.round(prediction.confidence * 100) : 0
  
  return (
    <div className="container">
      <h1>Parkinson's Disease Detection</h1>
      <p className="description">
        Upload an image to detect signs of Parkinson's disease
      </p>
      
      <div className="upload-section">
        <label className="file-input-label">
          <input 
            type="file" 
            accept="image/*" 
            onChange={handleFileSelect} 
            className="file-input" 
          />
          Choose Image
        </label>
        
        {preview && (
          <div className="preview-container">
            <img src={preview} alt="Preview" className="image-preview" />
            <button 
              onClick={handleSubmit} 
              className="analyze-button"
              disabled={loading}
            >
              {loading ? "Analyzing..." : "Analyze Image"}
            </button>
          </div>
        )}
      </div>
      
      {error && <div className="error-message">{error}</div>}
      
      {prediction && (
        <div className={`result-card ${prediction.prediction.includes("Parkinson") ? "positive" : "negative"}`}>
          <h2>Result</h2>
          <div className="result-content">
            <p className="prediction">{prediction.prediction}</p>
            <div className="confidence-bar-container">
              <div className="confidence-label">Confidence: {confidencePercent}%</div>
              <div className="confidence-bar">
                <div 
                  className="confidence-fill" 
                  style={{ width: `${confidencePercent}%` }}
                ></div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
