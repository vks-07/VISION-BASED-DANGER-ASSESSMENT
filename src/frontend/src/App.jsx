import { useState } from 'react'
import './App.css'

function App() {
  const [file, setFile] = useState(null)
  const [description, setDescription] = useState("")
  const [score, setScore] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handlePredict = async (e) => {
    e.preventDefault()
    if (!file || !description) {
      setError("Please provide both a video file and a context description.")
      return
    }

    setLoading(true)
    setError(null)
    setScore(null)

    // Append file and text into a FormData object to send to FastAPI
    const formData = new FormData()
    formData.append("video", file)
    formData.append("description", description)

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData
      })
      
      const data = await response.json()
      if (data.status === "success") {
        setScore(data.danger_score)
      } else {
        setError("Failed to calculate score. Backend may have encountered an error.")
      }
    } catch (err) {
      setError("Server connection failed. Make sure the FastAPI backend is running!")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <header className="header">
        <h1>🚨 ViDAS Danger Score</h1>
        <p>AI-Powered Video Risk Assessment</p>
      </header>
      
      <main className="main-content">
        <form onSubmit={handlePredict} className="upload-box">
          <div className="form-group">
            <label>Upload Video Segment (.mp4)</label>
            <input 
              type="file" 
              accept="video/mp4,video/*"
              onChange={(e) => setFile(e.target.files[0])} 
            />
          </div>
          
          <div className="form-group">
            <label>Context / Description</label>
            <textarea 
              rows="3" 
              placeholder="e.g. A person skateboarding recklessly down a steep hill into heavy traffic..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
            ></textarea>
          </div>

          <button type="submit" disabled={loading} className={loading ? "loading-btn" : ""}>
            {loading ? "Analyzing Video (This takes a minute)..." : "Analyze Danger Score"}
          </button>
        </form>

        {error && <div className="error-message">{error}</div>}

        {score !== null && (
          <div className="result-card">
            <h2>Predicted Danger Score</h2>
            <div className="score-display">
              <span className="score-number">{score.toFixed(2)}</span>
              <span className="score-max">/ 10.0</span>
            </div>
            
            <div className="progress-bar-bg">
              <div 
                className="progress-bar-fill" 
                style={{ 
                  width: `${(score / 10.0) * 100}%`, 
                  backgroundColor: score >= 7.0 ? '#ff4757' : (score >= 4.0 ? '#ffa502' : '#2ed573') 
                }}
              ></div>
            </div>
            
            <p className="score-label">
              {score >= 7.0 ? 'HIGH DANGER ⚠️' : (score >= 4.0 ? 'MODERATE RISK ⚠️' : 'LOW RISK ✅')}
            </p>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
