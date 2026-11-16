"""NOAA SRS region lookup shared with the STEREO frame maker."""
from pathlib import Path
from datetime import datetime as dt,timedelta
import json,re,urllib.request
import pandas as pd

def find_NOAA_location(number,requested,cache_dir,local_file=None,max_age=36):
	"""Return Stonyhurst lon, lat, validity epoch and NOAA SRS provenance.

	local_file may be a file or directory. It is read first; if it is missing,
	empty or has no report within max_age hours, NOAA SRS reports are downloaded
	into that folder (or beside the named file). local_file=None downloads into
	cache_dir. AR_location.json (the chosen report row) is saved beside the
	reports used; cache_dir is only created when local_file is None.
	"""
	if number is None or int(number)!=number or int(number)<=0:
		raise ValueError('Set NOAA_number to the NOAA active-region number')
	number=int(number)
	target=dt.fromisoformat(requested.replace('Z','+00:00'))
	if target.tzinfo is not None:
		target=pd.Timestamp(target).tz_convert('UTC').tz_localize(None).to_pydatetime()
	cache=Path(cache_dir)
	reports=[];failures=[]

	def online(folder):
		"""NOAA daily SRS for the day before, of and after the target, saved in folder."""
		folder.mkdir(parents=True,exist_ok=True)
		out=[]
		for offset in (-1,0,1):
			day=target+timedelta(days=offset)
			url=f'https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/solar_region_summaries/{day:%Y/%m/%Y%m%d}SRS.txt'
			out.append((folder/f'{day:%Y%m%d}_000000_SRS.txt',url))
		return out

	def read(candidates):
		for path,url in candidates:
			try:
				if not path.is_file():
					if url is None:raise FileNotFoundError(path)
					with urllib.request.urlopen(url,timeout=30) as response:
						text=response.read().decode('utf-8',errors='replace')
					if 'Solar Region Summary' not in text and 'SOLAR REGION SUMMARY' not in text:
						raise ValueError('Response is not a NOAA SRS report')
					path.write_text(text)
				text=path.read_text()
				issued=re.search(r':Issued:\s*(\d{4}\s+\w{3}\s+\d{1,2}\s+\d{4})\s+UTC',text,re.I)
				if issued:issue=dt.strptime(issued[1],'%Y %b %d %H%M')
				else:
					issued=re.search(r'Issued at (\d{4})Z on (\d{1,2} \w{3} \d{4})',text,re.I)
					if issued is None:raise ValueError('Missing issue date in SRS header')
					issue=dt.strptime(issued[2]+' '+issued[1],'%d %b %Y %H%M')
				valid=re.search(r'(?:Locations|Positions) Valid at\s+(\d{2})/(\d{2})(\d{2})Z',text,re.I)
				if valid is None:raise ValueError('Missing location-validity epoch in SRS header')
				dd,hh,mm=map(int,valid.groups())
				if not (0<=hh<=24 and 0<=mm<60 and (hh<24 or mm==0)):
					raise ValueError('Invalid SRS validity time')
				epochs=[]
				for shift in (-2,-1,0,1):
					candidate=issue.replace(hour=0,minute=0,second=0,microsecond=0)+timedelta(days=shift)
					if candidate.day==dd:epochs.append(candidate+timedelta(hours=hh,minutes=mm))
				if not epochs:raise ValueError('Cannot resolve SRS validity date')
				epoch=min(epochs,key=lambda t:abs((t-issue).total_seconds()))
				disk=re.split(r'^\s*II\.',text,flags=re.M)[0]
				for match in re.finditer(r'^\s*(\d{4,5})\s+([NS])(\d{2})([EW])(\d{2})\b',disk,re.M):
					n=int(match[1])
					if n!=number and not (n<10000 and number>=10000 and n==number%10000):continue
					latitude=int(match[3])*(1 if match[2]=='N' else -1)
					longitude=int(match[5])*(1 if match[4]=='W' else -1)
					age=abs((epoch-target).total_seconds())/3600
					if age<=max_age:
						reports.append(dict(noaa=number,lon=longitude,lat=latitude,epoch=epoch.isoformat(),issue=issue.isoformat(),offset_hours=age,file=str(path.resolve()),source=url or str(path.resolve())))
			except (OSError,ValueError) as exc:
				failures.append(f'{path.name}: {exc}')

	local=Path(local_file).expanduser() if local_file is not None else None
	if local is None:
		read(online(cache));record=cache
	else:
		if local.is_dir():
			paths=sorted(p for p in local.rglob('*') if p.is_file() and 'srs' in p.name.lower() and p.suffix.lower()=='.txt')
		else:
			paths=[local] if local.is_file() else []
		read([(path,None) for path in paths])
		record=local if (local.is_dir() or not local.suffix) else local.parent
		if not reports:
			# Missing/empty folder or no report near the target: fetch from NOAA,
			# saving into the SRS folder (or beside a named file) for next time.
			why='no SRS reports' if not paths else f'no report for NOAA {number} within {max_age} h'
			folder=local if (local.is_dir() or not local.suffix) else local.parent
			print(f'SRS_file {local}: {why}; downloading NOAA SRS into {folder}',flush=True)
			read(online(folder))
	if not reports:
		raise ValueError(f'NOAA {number}: no disk location within {max_age} h of {requested}. Check the number/date or supply SRS_file. '+'; '.join(failures))
	best=min(reports,key=lambda row:row['offset_hours'])
	record.mkdir(parents=True,exist_ok=True)  # beside the SRS reports used
	(record/'AR_location.json').write_text(json.dumps(best,indent=2))
	print(f'NOAA {number}: lon={best["lon"]:+g}, lat={best["lat"]:+g}; valid {best["epoch"]} UT')
	return best['lon'],best['lat'],best['epoch'],best
