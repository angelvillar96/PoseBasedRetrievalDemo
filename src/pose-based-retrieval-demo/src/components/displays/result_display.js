import React from "react"

import Text from '../titles/text_display'
import "./styles/display_styles.css"

class ResultDisplay extends React.Component{

  render(){
    var title = "Retrieval #" + this.props.det_idx + "%aux%" +
                "Metric: " + this.props.metric
    var img = URL.createObjectURL(this.props.file)
    return(
      <div className="img_display_area result">
        <div>
          <Text text_display={title} font_size="1.2em" font_style="normal"/>
        </div>
        <div className="result_display" style={{backgroundImage: "url("+img+")"}}></div>
      </div>
    )
  }

}

export default ResultDisplay

//
